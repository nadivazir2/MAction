
import os
import json
import time
import shutil
import hashlib
import tarfile
from pathlib import Path

import requests
from huggingface_hub import snapshot_download


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_ID = os.environ["MODEL_ID"]
REVISION = os.environ["MODEL_REVISION"]
MODEL_KEY = os.environ["MODEL_KEY"]

HF_TOKEN = os.getenv("HF_TOKEN") or None
BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

PART_SIZE = int(os.getenv("PART_MIB", "45")) * 1024 * 1024
MAX_WORKERS = int(os.getenv("OMNIVOICE_DOWNLOAD_MAX_WORKERS", "8"))

WORK = Path("work")
CACHE = WORK / "hf_cache"
PACKAGE = WORK / "package"
OUTPUT = WORK / "output"

WORK.mkdir(exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
PACKAGE.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def split_file(source, part_size):
    parts = []
    index = 1

    with open(source, "rb") as src:
        while True:
            data = src.read(part_size)
            if not data:
                break

            part = OUTPUT / f"{source.name}.part{index:04d}"
            with open(part, "wb") as dst:
                dst.write(data)

            parts.append(part)
            print(f"Created {part.name}: {part.stat().st_size:,} bytes")
            index += 1

    return parts


def telegram_upload(path, caption):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument"

    for attempt in range(1, 6):
        try:
            with open(path, "rb") as f:
                response = requests.post(
                    url,
                    data={
                        "chat_id": CHAT_ID,
                        "caption": caption[:1000],
                    },
                    files={
                        "document": (
                            path.name,
                            f,
                            "application/octet-stream",
                        )
                    },
                    timeout=(30, 900),
                )

            response.raise_for_status()
            result = response.json()

            if not result.get("ok"):
                raise RuntimeError(str(result))

            print(f"Telegram uploaded: {path.name}")
            return

        except Exception as exc:
            print(f"Upload attempt {attempt}/5 failed: {exc}")

            if attempt == 5:
                raise

            time.sleep(min(30 * attempt, 120))


# --------------------------------------------------
# 1. Download exact VoiceStudio revision
# --------------------------------------------------

print(f"Downloading {MODEL_ID}")
print(f"Revision: {REVISION}")
print(f"Endpoint: {os.getenv('HF_ENDPOINT')}")

snapshot_path = Path(
    snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        cache_dir=str(CACHE),
        token=HF_TOKEN,
        max_workers=MAX_WORKERS,
    )
)

if snapshot_path.name != REVISION:
    raise RuntimeError(
        f"Unexpected snapshot revision: {snapshot_path.name}"
    )

if not snapshot_path.is_dir():
    raise RuntimeError("Snapshot directory not found")

files = [p for p in snapshot_path.rglob("*") if p.is_file()]

if not files:
    raise RuntimeError("Downloaded snapshot is empty")

print(f"Downloaded files: {len(files)}")


# --------------------------------------------------
# 2. Create Windows-friendly HF cache
# --------------------------------------------------

repo_cache_name = "models--" + MODEL_ID.replace("/", "--")

source_repo = CACHE / repo_cache_name
export_repo = PACKAGE / repo_cache_name

if not source_repo.is_dir():
    raise RuntimeError("HF repository cache not found")

# Copy the complete HF repository cache.
# symlinks=False resolves snapshot links into regular files.
# This avoids Windows symlink permission problems.
shutil.copytree(
    source_repo,
    export_repo,
    symlinks=False,
    dirs_exist_ok=True,
)

# Verify all snapshot files exist as regular files.
export_snapshot = export_repo / "snapshots" / REVISION

if not export_snapshot.is_dir():
    raise RuntimeError("Exported snapshot missing")

for path in export_snapshot.rglob("*"):
    if path.is_symlink():
        raise RuntimeError(f"Unexpected symlink: {path}")

    if path.is_file() and path.stat().st_size == 0:
        # Empty files are allowed only if they were empty upstream.
        original = snapshot_path / path.relative_to(export_snapshot)
        if original.stat().st_size != 0:
            raise RuntimeError(f"Unexpected empty file: {path}")

# Preserve the branch reference and VoiceStudio revision marker.
refs = export_repo / "refs"
refs.mkdir(parents=True, exist_ok=True)
(refs / "main").write_text(REVISION + "\n", encoding="ascii")

(export_repo / "voicestudio-revision").write_text(
    REVISION + "\n",
    encoding="ascii",
)

print("Windows-compatible cache prepared.")


# --------------------------------------------------
# 3. Create archive
# --------------------------------------------------

archive = OUTPUT / f"{MODEL_KEY}_{REVISION}.tar"

with tarfile.open(
    archive,
    mode="w",
    format=tarfile.PAX_FORMAT,
) as tar:
    tar.add(export_repo, arcname=repo_cache_name)

archive_size = archive.stat().st_size
archive_hash = sha256_file(archive)

print(f"Archive size: {archive_size:,} bytes")
print(f"Archive SHA256: {archive_hash}")


# --------------------------------------------------
# 4. Split archive
# --------------------------------------------------

parts = split_file(archive, PART_SIZE)

if not parts:
    raise RuntimeError("No archive parts created")

part_info = []

for part in parts:
    part_info.append({
        "name": part.name,
        "size": part.stat().st_size,
        "sha256": sha256_file(part),
    })

manifest = {
    "model_key": MODEL_KEY,
    "repo_id": MODEL_ID,
    "revision": REVISION,
    "cache_folder": repo_cache_name,
    "archive": archive.name,
    "archive_size": archive_size,
    "archive_sha256": archive_hash,
    "part_size_bytes": PART_SIZE,
    "parts": part_info,
    "snapshot_file_count": len(files),
    "hf_endpoint": os.getenv("HF_ENDPOINT"),
}

manifest_path = OUTPUT / f"{MODEL_KEY}_manifest.json"

manifest_path.write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)


# --------------------------------------------------
# 5. Upload to Telegram
# --------------------------------------------------

total = len(parts)

for index, part in enumerate(parts, start=1):
    caption = (
        f"VoiceStudio Model: {MODEL_KEY}\n"
        f"Revision: {REVISION}\n"
        f"Part: {index}/{total}"
    )

    telegram_upload(part, caption)

telegram_upload(
    manifest_path,
    f"VoiceStudio manifest: {MODEL_KEY}",
)

print(f"SUCCESS: {MODEL_KEY} fully uploaded.")
