param([int]$procid = 0, [string]$out = "$PSScriptRoot\..\..\runs\shot.png")
Add-Type -AssemblyName System.Drawing
Add-Type @'
using System; using System.Runtime.InteropServices;
public class W {
 [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
 [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
 [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
 public struct RECT { public int L,T,R,B; } }
'@
[W]::SetProcessDPIAware() | Out-Null
$h = $(if ($procid) { (Get-Process -Id $procid).MainWindowHandle } else { (Get-Process hoi4 | Select-Object -First 1).MainWindowHandle })
[W]::SetForegroundWindow($h) | Out-Null; Start-Sleep -Milliseconds 500
$r = New-Object W+RECT; [W]::GetWindowRect($h, [ref]$r) | Out-Null
$w = $r.R - $r.L; $ht = $r.B - $r.T
$bmp = New-Object Drawing.Bitmap $w, $ht
[Drawing.Graphics]::FromImage($bmp).CopyFromScreen($r.L, $r.T, 0, 0, $bmp.Size)
$small = New-Object Drawing.Bitmap $bmp, ([int]($w/2)), ([int]($ht/2))
$small.Save($out)
"$($r.L),$($r.T) ${w}x${ht}"
