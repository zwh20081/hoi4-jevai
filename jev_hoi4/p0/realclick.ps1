# Bring hoi4 to the foreground and send a real (SendInput) left click at window-relative pixel (x, y), N times.
param([int]$procid = 0, [int]$x, [int]$y, [int]$n = 1)
Add-Type @'
using System; using System.Runtime.InteropServices; using System.Threading;
public class RC {
 [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
 [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
 [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int c);
 [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
 [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
 [DllImport("user32.dll")] public static extern void mouse_event(uint f, int dx, int dy, uint d, IntPtr e);
 public struct RECT { public int L,T,R,B; }
 public static void Click(IntPtr h, int wx, int wy) {
   RECT r; GetWindowRect(h, out r); SetCursorPos(r.L + wx, r.T + wy); Thread.Sleep(120);
   mouse_event(2, 0, 0, 0, IntPtr.Zero); Thread.Sleep(70); mouse_event(4, 0, 0, 0, IntPtr.Zero); Thread.Sleep(250); } }
'@
[RC]::SetProcessDPIAware() | Out-Null
$h = $(if ($procid) { (Get-Process -Id $procid).MainWindowHandle } else { (Get-Process hoi4 | Select-Object -First 1).MainWindowHandle })
[RC]::ShowWindow($h, 9) | Out-Null; [RC]::SetForegroundWindow($h) | Out-Null; Start-Sleep -Milliseconds 400
for ($i = 0; $i -lt $n; $i++) { [RC]::Click($h, $x, $y) }
