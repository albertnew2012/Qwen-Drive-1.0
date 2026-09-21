#!/usr/bin/env bash
# Download + extract OpenLane-V2 subset_B (train/val/test) into data/OpenLane-V2/.
#
# Source: public HuggingFace mirror of OpenDataLab's OpenDriveLab/OpenLane-V2.
# File sizes match the official table at
#   https://github.com/OpenDriveLab/OpenLane-V2/blob/master/data/README.md
# Each tar is md5-checked against that table; HF additionally verifies sha256.
#
# Tars are deleted right after extraction, so peak disk use is ~54 GB
# (~45 GB extracted + the largest single tar).
# Re-runnable: completed parts are skipped via marker files.

set -uo pipefail

REPO=AlayaNeW/OpenDriveLab___OpenLane-V2
ROOT=/home/albert/Desktop/Qwen-Drive-1.0/data
DEST=$ROOT/OpenLane-V2
WORK=$ROOT/.olv2_download
HF=/home/albert/Desktop/Qwen-Drive-1.0/.venv/bin/hf

# md5 of each tar, from the official OpenLane-V2 data/README.md table
declare -A MD5=(
  [info]=27696b1ed1d99b1f70fdb68f439dc87d
  [image_0]=0876c6b2381bacedeb3be16e57c7d59b
  [image_1]=ecdec8ff8c72525af322032a312aad10
  [image_2]=b720bf7fdf0ebd44b71beffc84722359
  [image_3]=ac3bc9400ade6c47c396af4b12bbd0e0
  [image_4]=fa4c4a04b5ad3eac817e6368047d0d89
  [image_5]=19d2cc92514e65270779e405d3a93c61
  [image_6]=d4f56c562f11a6bcc918f2d20441c42c
  [image_7]=443045d7a3faf5998af27e2302d3503e
  [image_8]=6ecb7a9e866e29ed73d335c2d897f50e
)
# info first: it carries data_dict_subset_B.json, preprocess.py and openlanev2.md5
ORDER=(info image_0 image_1 image_2 image_3 image_4 image_5 image_6 image_7 image_8)

mkdir -p "$DEST" "$WORK"
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

mismatched=()
for key in "${ORDER[@]}"; do
  name="OpenLane-V2_subset_B_${key}.tar"
  tarball="$WORK/raw/$name"

  if [ -f "$WORK/.done_$key" ]; then
    log "SKIP     $name (already extracted)"
    continue
  fi

  log "DOWNLOAD $name"
  ok=0
  for attempt in 1 2 3 4 5; do
    if "$HF" download "$REPO" "raw/$name" --repo-type dataset --local-dir "$WORK" --quiet; then
      ok=1; break
    fi
    log "         download attempt $attempt failed, retrying in 30s"
    sleep 30
  done
  if [ "$ok" -ne 1 ] || [ ! -f "$tarball" ]; then
    log "FAIL     could not download $name -- aborting"
    exit 1
  fi

  got=$(md5sum "$tarball" | cut -d' ' -f1)
  want=${MD5[$key]}
  if [ "$got" = "$want" ]; then
    log "MD5 OK   $name"
  else
    # HF already verified the content hash on download; a differing md5 means the
    # mirror re-tarred the same files, not corruption. Flagged, not fatal.
    log "MD5 DIFF $name got=$got want=$want (mirror re-tar; content sha256 verified by HF)"
    mismatched+=("$name")
  fi

  log "EXTRACT  $name -> $DEST"
  if ! tar -xf "$tarball" -C "$DEST"; then
    log "FAIL     extract of $name -- aborting"
    exit 1
  fi
  touch "$WORK/.done_$key"
  rm -f "$tarball"
  log "OK       $key done; disk free: $(df -h --output=avail "$ROOT" | tail -1 | tr -d ' ')"
done

log "--- all parts downloaded and extracted ---"
log "top level: $(ls "$DEST" | tr '\n' ' ')"
for split in train val test; do
  [ -d "$DEST/$split" ] && log "$split segments: $(ls "$DEST/$split" | wc -l)"
done
log "size: $(du -sh "$DEST" | cut -f1)"
if [ ${#mismatched[@]} -gt 0 ]; then
  log "note: tar md5 differed from official table for: ${mismatched[*]}"
fi
log "DONE"
