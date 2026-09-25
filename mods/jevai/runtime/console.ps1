# Type commands into the HOI4 console. Brings hoi4 to the foreground and uses SendInput
# (posted WM_CHAR/WM_KEYDOWN are ignored by the game's SDL input layer, measured 2026-09-23).
# The console is toggled with the ` key (scan 0x29); text goes in as KEYEVENTF_UNICODE.
# -cmds takes one string with commands separated by ';' (arrays do not survive a bash -> powershell call).
# Several processes (collector, jevd) type into different instances: a named mutex serialises them, and
# the target window must actually be foreground before any key is sent (exit code 2 otherwise).
param([int]$procid = 0, [string]$cmds = "", [switch]$noToggle, [switch]$open)
Add-Type @'
using System; using System.Runtime.InteropServices; using System.Threading;
public class SI {
 [StructLayout(LayoutKind.Sequential)] struct KEYBDINPUT { public ushort vk, scan; public uint flags, time; public IntPtr extra; }
 [StructLayout(LayoutKind.Explicit, Size = 40)] struct INPUT { [FieldOffset(0)] public uint type; [FieldOffset(8)] public KEYBDINPUT ki; }
 [DllImport("user32.dll")] static extern uint SendInput(uint n, INPUT[] i, int size);
 [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
 [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
 [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int c);
 [DllImport("user32.dll")] public static extern bool BringWindowToTop(IntPtr h);
 [DllImport("user32.dll")] public static extern void keybd_event(byte vk, byte scan, uint f, IntPtr e);
 static void Send(ushort vk, ushort scan, uint flags) {
   INPUT[] a = new INPUT[1]; a[0].type = 1; a[0].ki.vk = vk; a[0].ki.scan = scan; a[0].ki.flags = flags;
   SendInput(1, a, Marshal.SizeOf(typeof(INPUT))); Thread.Sleep(12); }
 public static void Scan(ushort scan) { Send(0, scan, 8); Thread.Sleep(30); Send(0, scan, 8 | 2); Thread.Sleep(30); }
 public static void Text(string t) { foreach (char c in t) { Send(0, c, 4); Send(0, c, 4 | 2); } }
 // Windows only lets the process that received the last input steal focus; a synthetic Alt tap counts.
 public static bool Focus(IntPtr h) {
   for (int i = 0; i < 6; i++) {
     if (GetForegroundWindow() == h) return true;
     keybd_event(0x12, 0, 0, IntPtr.Zero); keybd_event(0x12, 0, 2, IntPtr.Zero);
     ShowWindow(h, 9); BringWindowToTop(h); SetForegroundWindow(h); Thread.Sleep(250);
   }
   return GetForegroundWindow() == h;
 }
}
'@
$h = $(if ($procid) { (Get-Process -Id $procid).MainWindowHandle } else { (Get-Process hoi4 | Select-Object -First 1).MainWindowHandle })
$mtx = New-Object System.Threading.Mutex($false, "Global\JevAIConsoleInput")
[void]$mtx.WaitOne(120000)
try {
  if (-not [SI]::Focus($h)) { Write-Output "focus failed"; exit 2 }
  # a window that has just been raised drops the first keys it gets; give it time to settle
  Start-Sleep -Milliseconds 900
  if (-not $noToggle -and -not $open) { [SI]::Scan(0x29); Start-Sleep -Milliseconds 700 }
  foreach ($c in ($cmds -split ';' | ForEach-Object { $_.Trim() } | Where-Object { $_ })) {
    if ([SI]::GetForegroundWindow() -ne $h) { Write-Output "lost focus"; exit 2 }
    [SI]::Text($c); Start-Sleep -Milliseconds 100; [SI]::Scan(0x1C); Start-Sleep -Milliseconds 400
  }
  if (-not $noToggle) { [SI]::Scan(0x29) }
} finally { $mtx.ReleaseMutex() }
