# Drive the front-end from the main menu into a 1936 game (Germany selected by default), then
# open the console and run the given commands. Coordinates are window pixels at 1944x1139 (1920x1080 client).
param([string]$cmds = "observe;gamespeed 5", [int]$loadWait = 40)
$here = $PSScriptRoot
function Click($x, $y, $wait) { & powershell -ExecutionPolicy Bypass -File "$here\realclick.ps1" -x $x -y $y; Start-Sleep -Seconds $wait }
Click 972 211 6      # Single Player
Click 972 407 8      # New Game
Click 1194 913 10    # Select Scenario (1 January 1936)
Click 1096 1090 8    # Select Country
Click 1718 1064 $loadWait  # START
& powershell -ExecutionPolicy Bypass -File "$here\console.ps1" -cmds $cmds
