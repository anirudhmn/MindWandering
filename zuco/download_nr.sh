#!/bin/zsh
# Download ZuCo Task2 (Normal Reading, Wikipedia) Matlab files with resume.
cd "$(dirname "$0")"
mkdir -p task2_NR_matlab
LOG=download_nr.log
echo "=== start $(date) ===" >> "$LOG"
while IFS=$'\t' read -r name url; do
  out="task2_NR_matlab/$name"
  echo "--- $name from $url ---" >> "$LOG"
  # -C - resume; -L follow redirects; --retry for transient failures
  curl -L -C - --retry 5 --retry-delay 5 --retry-all-errors \
       -o "$out" "$url" >> "$LOG" 2>&1
  echo "done $name size=$(stat -f%z "$out" 2>/dev/null)" >> "$LOG"
done < nr_matlab_downloads.tsv
echo "=== ALL DONE $(date) ===" >> "$LOG"
