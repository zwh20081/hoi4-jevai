# Move a hoi4 window so several instances do not overlap: -procid <pid> -x <left> -y <top>
param([int]$procid, [int]$x, [int]$y)
Add-Type @'
using System; using System.Runtime.InteropServices;
public class MW { [DllImport("user32.dll")] public static extern bool SetWindowPos(IntPtr h, IntPtr a, int x, int y, int cx, int cy, uint f); }
'@
[MW]::SetWindowPos((Get-Process -Id $procid).MainWindowHandle, [IntPtr]::Zero, $x, $y, 0, 0, 0x0001 -bor 0x0004) | Out-Null
