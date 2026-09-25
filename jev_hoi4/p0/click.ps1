# Post a left click to the hoi4 window at window-relative pixel (x, y), N times. No focus change.
param([int]$x, [int]$y, [int]$n = 1)
Add-Type @'
using System; using System.Runtime.InteropServices; using System.Threading;
public class PC {
 [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
 [DllImport("user32.dll")] static extern bool PostMessage(IntPtr h, uint m, IntPtr w, IntPtr l);
 [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
 [DllImport("user32.dll")] public static extern bool ScreenToClient(IntPtr h, ref POINT p);
 public struct RECT { public int L,T,R,B; } public struct POINT { public int X,Y; }
 public static void Click(IntPtr h, int wx, int wy) {
   RECT r; GetWindowRect(h, out r); POINT p; p.X = r.L + wx; p.Y = r.T + wy; ScreenToClient(h, ref p);
   IntPtr l = (IntPtr)((p.Y << 16) | (p.X & 0xFFFF));
   PostMessage(h, 0x200, IntPtr.Zero, l); Thread.Sleep(60);
   PostMessage(h, 0x201, (IntPtr)1, l); Thread.Sleep(60); PostMessage(h, 0x202, IntPtr.Zero, l); Thread.Sleep(150); } }
'@
[PC]::SetProcessDPIAware() | Out-Null
$h = (Get-Process hoi4).MainWindowHandle
for ($i = 0; $i -lt $n; $i++) { [PC]::Click($h, $x, $y) }
