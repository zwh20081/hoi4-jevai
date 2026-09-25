# Launch HOI4 against a userdir. Extra game args pass through, ';'-separated, e.g.
#   powershell -File launch.ps1 -gameArgs "-ogl;-dump_history" -userdir C:\jevai\u1 -pidfile temp\inst\u1.pid
param([string]$gameArgs = "", [string]$userdir = "", [string]$pidfile = "")
$root = (Resolve-Path "$PSScriptRoot\..\..").Path
$game = if ($env:HOI4_EXE) { $env:HOI4_EXE } else { "C:\Users\zwh20081\Desktop\h4\hoi4.exe" }  # the game install is outside the project
if (-not $userdir) { $userdir = "temp\hoi4user" }
if (-not [IO.Path]::IsPathRooted($userdir)) { $userdir = Join-Path $root $userdir }
$user = (Resolve-Path $userdir).Path
if (-not $pidfile) { $pidfile = "$root\temp\hoi4.pid" }
$a = @("-userdir=$user", "-quickstart", "-nolauncher") + @($gameArgs -split ';' | Where-Object { $_ })
$p = Start-Process -FilePath $game -WorkingDirectory (Split-Path $game) -ArgumentList $a -PassThru
# below normal: the games still get every idle cycle, but the desktop stays responsive
try { $p.PriorityClass = 'BelowNormal' } catch {}
$p.Id | Out-File -Encoding ascii $pidfile
"started pid $($p.Id): $($a -join ' ')"
