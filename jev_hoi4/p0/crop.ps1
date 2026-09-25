# Full-resolution crop of the hoi4 window: -x -y -w -h in window pixels.
param([int]$procid = 0, [int]$x = 0, [int]$y = 0, [int]$w = 600, [int]$h = 200, [string]$out = "$PSScriptRoot\..\..\runs\crop.png")
Add-Type -AssemblyName System.Drawing
Add-Type @'
using System; using System.Runtime.InteropServices;
public class W2 { [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
 [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
 public struct RECT { public int L,T,R,B; } }
'@
[W2]::SetProcessDPIAware() | Out-Null
$hw = $(if ($procid) { (Get-Process -Id $procid).MainWindowHandle } else { (Get-Process hoi4 | Select-Object -First 1).MainWindowHandle })
$r = New-Object W2+RECT; [W2]::GetWindowRect($hw, [ref]$r) | Out-Null
$bmp = New-Object Drawing.Bitmap $w, $h
[Drawing.Graphics]::FromImage($bmp).CopyFromScreen($r.L + $x, $r.T + $y, 0, 0, $bmp.Size)
$bmp.Save($out)
