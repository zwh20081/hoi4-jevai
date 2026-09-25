#!/bin/bash
# Run several observer games back to back (vanilla AI, no intervention).
#   bash jev_hoi4/collect_all.sh
cd "$(dirname "$0")/.."
while read -r run save seed end; do
  [ -z "$run" ] && continue
  [ -f "runs/games/$run/meta.json" ] && { echo "skip $run"; continue; }
  bash jev_hoi4/collect.sh "$run" "$save" "$seed" "$end"
done <<'EOF'
g03 jev_1936_06.hoi4 30303 1944
g04 jev_1936_06_nonhist.hoi4 40404 1944
g05 jev_1936_06.hoi4 50505 1944
EOF
