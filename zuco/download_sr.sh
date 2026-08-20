#!/bin/zsh
cd "$(dirname "$0")"; mkdir -p task1_SR_matlab; LOG=download_sr.log
echo "=== start $(date) ===" >> "$LOG"
while IFS=$'\t' read -r name url; do
  out="task1_SR_matlab/$name"
  curl -L -C - --retry 6 --retry-delay 5 --retry-all-errors -o "$out" "$url" >> "$LOG" 2>&1
  echo "done $name size=$(stat -f%z "$out" 2>/dev/null)" >> "$LOG"
done < sr_matlab_downloads.tsv
echo "=== ALL DONE $(date) ===" >> "$LOG"
