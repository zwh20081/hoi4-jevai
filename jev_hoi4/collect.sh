#!/bin/bash
# Collect observer games: vanilla AI plays, -dump_history records every month.
#   bash jev_hoi4/collect.sh <run_id> <save_file> <seed> [end_year]
# Each run: launch -> load save -> observe, reseed, speed 5 -> wait for end year / crash / stall -> archive dump.
set -u
cd "$(dirname "$0")/.."
RUN=$1; SAVE=$2; SEED=$3; END=${4:-1946}
U=hoi4user; OUT=runs/games/$RUN
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $RUN: $*" | tee -a runs/collect.log; }

taskkill //IM hoi4.exe //F >/dev/null 2>&1; sleep 4
rm -rf $U/history_dump "$U/save games/autosave.hoi4"
touch "$U/save games/$SAVE"
log "launch save=$SAVE seed=$SEED end=$END"
powershell -ExecutionPolicy Bypass -File jev_hoi4/p0/launch.ps1 -gameArgs "-debug;-dump_history" >/dev/null
# wait for main menu (game.log starts being written once data is loaded)
for i in $(seq 1 60); do sleep 3; grep -aq "Executing History" $U/logs/game.log 2>/dev/null && break; done
sleep 8
powershell -ExecutionPolicy Bypass -File jev_hoi4/p0/loadgame.ps1 -cmds "" -loadWait 5 >/dev/null
# the front-end log is stamped 1936.01.01; once the save is in, lines carry the save date (paused)
for i in $(seq 1 60); do
  sleep 3
  grep -a -o '\]\[19[0-9][0-9]\.[0-9.]*\]' $U/logs/game.log | tail -1 | grep -qv '1936\.01\.01' && break
done
sleep 5
powershell -ExecutionPolicy Bypass -File jev_hoi4/p0/console.ps1 -cmds "observe;random_seed $SEED;debug_norender;debug_nogui;gamespeed 5" >/dev/null
d0=$(grep -a -o '\]\[19[0-9][0-9]\.[0-9.]*\]' $U/logs/game.log | tail -1)
for i in $(seq 1 8); do
  sleep 15
  d1=$(grep -a -o '\]\[19[0-9][0-9]\.[0-9.]*\]' $U/logs/game.log | tail -1)
  [ "$d1" != "$d0" ] && break
  # clock not moving: the command line was probably lost; only gamespeed is safe to resend (the rest are toggles)
  powershell -ExecutionPolicy Bypass -File jev_hoi4/p0/console.ps1 -cmds "gamespeed 5" >/dev/null
done
log "running from $d1"
last=""; still=0
while true; do
  sleep 30
  if ! tasklist | grep -qi hoi4.exe; then log "process gone (crash?)"; break; fi
  d=$(grep -a -o '\]\[19[0-9][0-9]\.[0-9.]*\]' $U/logs/game.log | tail -1 | tr -d '][')
  y=${d%%.*}
  if [ -n "$y" ] && [ "$y" -ge "$END" ]; then log "reached $d"; break; fi
  if [ "$d" = "$last" ]; then
    still=$((still+1))
    # stalled: probably paused or a popup grabbed input; re-issue speed once, then give up after ~5 min
    [ $still -eq 2 ] && powershell -ExecutionPolicy Bypass -File jev_hoi4/p0/console.ps1 -cmds "gamespeed 5" >/dev/null
    [ $still -ge 10 ] && { log "stalled at $d"; break; }
  else still=0; fi
  last=$d
done
taskkill //IM hoi4.exe //F >/dev/null 2>&1; sleep 4
n=$(ls $U/history_dump 2>/dev/null | grep -c '^[0-9]*\.txt$')
rm -rf "$OUT/history_dump"; cp -r $U/history_dump "$OUT/" 2>/dev/null
echo "{\"run\":\"$RUN\",\"save\":\"$SAVE\",\"seed\":$SEED,\"last_date\":\"$last\",\"months\":$n}" > "$OUT/meta.json"
log "archived $n months (last $last)"
