# From the main menu: Single Player -> Load -> first (newest) save -> Start, then console commands.
# Window pixels at 1944x1139. The load list is sorted newest-first, so touch the wanted save before launching.
param([int]$procid = 0, [string]$cmds = "observe;gamespeed 5", [int]$loadWait = 55)
$here = $PSScriptRoot
function Click($x, $y, $wait) { & powershell -ExecutionPolicy Bypass -File "$here\realclick.ps1" -procid $procid -x $x -y $y; Start-Sleep -Seconds $wait }
Click 972 211 5      # Single Player
Click 972 467 8      # Load
Click 200 390 3      # first save in the list
Click 1718 1064 $loadWait  # Start
if ($cmds) { & powershell -ExecutionPolicy Bypass -File "$here\console.ps1" -procid $procid -cmds $cmds }
