#!/usr/bin/env bash
# Download + extract the full nuScenes v1.0-trainval set (850 scenes) onto the
# 1.8T data disk.
#
# Source: the official Motional public bucket that nuscenes.org serves its
# download links from. Use of the data is governed by the nuScenes Terms of Use
# (non-commercial); register at https://www.nuscenes.org if you have not.
#
# Contents (294 GB of archives, ~293 GB extracted):
#   v1.0-trainval_meta.tgz          annotations + calibration json
#   v1.0-trainval{01..10}_blobs.tgz camera jpgs, lidar pcd, radar (samples + sweeps)
#   nuScenes-map-expansion-v1.3.zip vector map layers -> maps/
#   can_bus.zip                     vehicle CAN traces
#
# Archives are deleted right after extraction, so peak disk is ~293 GB extracted
# plus at most two staged tars (~70 GB).
#
# Integrity: no md5 list is published for these archives, so each download is
# checked byte-exact against the remote Content-Length, and gzip/zip CRCs are
# verified on extraction (a corrupt archive fails the extract and aborts).
#
# Re-runnable: finished parts are skipped via marker files, and partial
# downloads resume with HTTP range requests.

set -uo pipefail

URL=https://motional-nuscenes.s3.amazonaws.com/public/v1.0
DEST=/media/albert/932181ee-44ea-4d14-9a4f-2464ef28e809/nuscenes
WORK=$DEST/.download
LOG=$DEST/download.log

# meta first: it is small and makes the tree usable by the devkit immediately
FILES=(
  v1.0-trainval_meta.tgz
  v1.0-trainval01_blobs.tgz v1.0-trainval02_blobs.tgz v1.0-trainval03_blobs.tgz
  v1.0-trainval04_blobs.tgz v1.0-trainval05_blobs.tgz v1.0-trainval06_blobs.tgz
  v1.0-trainval07_blobs.tgz v1.0-trainval08_blobs.tgz v1.0-trainval09_blobs.tgz
  v1.0-trainval10_blobs.tgz
  nuScenes-map-expansion-v1.3.zip
  can_bus.zip
)

mkdir -p "$DEST" "$WORK"
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }

remote_size() {
  curl -sI -m 60 --retry 3 "$URL/$1" \
    | awk 'tolower($1)=="content-length:"{print $2+0}' | tail -1
}

local_size() { [ -f "$1" ] && stat -c%s "$1" || echo 0; }

# Download $1 into $WORK, resuming and retrying until it matches the remote
# byte count exactly. Safe to run in the background as a prefetch.
fetch() {
  local f=$1 dst="$WORK/$1" want have
  want=$(remote_size "$f")
  if [ -z "$want" ] || [ "$want" -eq 0 ]; then
    log "FAIL     cannot read remote size for $f"
    return 1
  fi
  for attempt in 1 2 3 4 5 6 7 8; do
    have=$(local_size "$dst")
    if [ "$have" -eq "$want" ]; then
      [ "$attempt" -gt 1 ] && log "         $f resumed to full size"
      return 0
    fi
    if [ "$have" -gt "$want" ]; then
      log "         $f staged file larger than remote; restarting it"
      rm -f "$dst"
    fi
    curl -fsS -m 0 --speed-limit 1024 --speed-time 120 -C - -o "$dst" "$URL/$f"
    [ "$(local_size "$dst")" -eq "$want" ] && return 0
    log "         $f attempt $attempt incomplete ($(local_size "$dst")/$want), retry in 20s"
    sleep 20
  done
  return 1
}

extract() {
  local f=$1 dst="$WORK/$1"
  case "$f" in
    nuScenes-map-expansion-*.zip)
      # basemap/ expansion/ prediction/ belong under maps/
      mkdir -p "$DEST/maps" && unzip -q -o "$dst" -d "$DEST/maps" ;;
    *.zip)
      unzip -q -o "$dst" -d "$DEST" ;;
    *.tgz)
      tar -xzf "$dst" -C "$DEST" ;;
  esac
}

PREFETCH_PID=""
PREFETCH_FILE=""

start_prefetch() {
  local f=${1:-}
  [ -z "$f" ] && return
  [ -f "$WORK/.done_$f" ] && return
  fetch "$f" >>"$LOG" 2>&1 &
  PREFETCH_PID=$!
  PREFETCH_FILE=$f
}

log "=== nuScenes v1.0-trainval download -> $DEST ==="
log "free space: $(df -h --output=avail "$DEST" | tail -1 | tr -d ' ')"

n=${#FILES[@]}
for ((i = 0; i < n; i++)); do
  f=${FILES[$i]}
  next=${FILES[$((i + 1))]:-}

  if [ -f "$WORK/.done_$f" ]; then
    log "SKIP     $f (already extracted)"
    continue
  fi

  if [ "$PREFETCH_FILE" = "$f" ] && [ -n "$PREFETCH_PID" ]; then
    log "DOWNLOAD $f (prefetching)"
    wait "$PREFETCH_PID"; rc=$?
    PREFETCH_PID=""; PREFETCH_FILE=""
    [ $rc -ne 0 ] && { log "FAIL     download of $f -- aborting"; exit 1; }
  else
    log "DOWNLOAD $f"
    fetch "$f" || { log "FAIL     download of $f -- aborting"; exit 1; }
  fi
  log "SIZE OK  $f ($(du -h "$WORK/$f" | cut -f1))"

  # pull the next archive while this one extracts; two streams roughly double
  # throughput on this link (~22 MB/s single, ~33 MB/s paired)
  start_prefetch "$next"

  log "EXTRACT  $f -> $DEST"
  if ! extract "$f"; then
    log "FAIL     extract of $f (archive corrupt) -- removing it so a re-run redownloads"
    rm -f "$WORK/$f"
    [ -n "$PREFETCH_PID" ] && kill "$PREFETCH_PID" 2>/dev/null
    exit 1
  fi
  touch "$WORK/.done_$f"
  rm -f "$WORK/$f"
  log "OK       $f done; disk free: $(df -h --output=avail "$DEST" | tail -1 | tr -d ' ')"
done

log "--- all archives downloaded and extracted ---"
log "top level: $(ls "$DEST" | tr '\n' ' ')"
for d in samples sweeps maps v1.0-trainval can_bus; do
  [ -d "$DEST/$d" ] && log "$d: $(du -sh "$DEST/$d" | cut -f1)"
done
[ -d "$DEST/v1.0-trainval" ] && log "meta json files: $(ls "$DEST/v1.0-trainval" | wc -l)"
log "total: $(du -sh "$DEST" | cut -f1)"
log "DONE"
