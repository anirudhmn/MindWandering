#!/bin/zsh
# Download ZuCo Task3 (Task-Specific Reading, Wikipedia) Matlab files with resume.
cd "$(dirname "$0")"
mkdir -p task3_TSR_matlab
LOG=download_tsr.log
echo "=== start $(date) ===" >> "$LOG"
while IFS=$'\t' read -r name url; do
  out="task3_TSR_matlab/$name"
  echo "--- $name from $url ---" >> "$LOG"
  curl -L -C - --retry 5 --retry-delay 5 --retry-all-errors \
       -o "$out" "$url" >> "$LOG" 2>&1
  echo "done $name size=$(stat -f%z "$out" 2>/dev/null)" >> "$LOG"
done < tsr_matlab_downloads.tsv
echo "=== ALL DONE $(date) ===" >> "$LOG"
