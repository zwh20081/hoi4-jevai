# Post a virtual key to the hoi4 window N times (no focus change).
param([int]$vk = 0x6B, [int]$n = 1)
Add-Type @'
using System; using System.Runtime.InteropServices; using System.Threading;
public class PK { [DllImport("user32.dll")] static extern bool PostMessage(IntPtr h, uint m, IntPtr w, IntPtr l);
 public static void Key(IntPtr h, int vk) { PostMessage(h, 0x100, (IntPtr)vk, (IntPtr)1); Thread.Sleep(40); PostMessage(h, 0x101, (IntPtr)vk, (IntPtr)(long)0xC0000001); Thread.Sleep(120); } }
'@
$h = (Get-Process hoi4).MainWindowHandle
for ($i = 0; $i -lt $n; $i++) { [PK]::Key($h, $vk) }
