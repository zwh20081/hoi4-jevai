Add-Type @'
using System; using System.Runtime.InteropServices;
public class WW { [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int c);
 [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h); }
'@
$h = (Get-Process hoi4).MainWindowHandle
[WW]::ShowWindow($h, 9) | Out-Null; Start-Sleep -Milliseconds 500; [WW]::SetForegroundWindow($h) | Out-Null
