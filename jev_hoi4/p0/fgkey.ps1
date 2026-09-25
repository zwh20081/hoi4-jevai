# Bring hoi4 to the foreground (real click on the title bar) and send one key with keybd_event.
param([int]$vk = 0xC0, [int]$scan = 0x29)
Add-Type @'
using System; using System.Runtime.InteropServices; using System.Threading;
public class K2 {
 [DllImport("user32.dll")] public static extern void keybd_event(byte vk, byte scan, uint f, IntPtr e);
 [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
 [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h); }
'@
$h = (Get-Process hoi4).MainWindowHandle
[K2]::SetForegroundWindow($h) | Out-Null; Start-Sleep -Milliseconds 300
"fg=$([K2]::GetForegroundWindow()) hoi4=$h"
[K2]::keybd_event($vk, $scan, 0, [IntPtr]::Zero); Start-Sleep -Milliseconds 60; [K2]::keybd_event($vk, $scan, 2, [IntPtr]::Zero)
