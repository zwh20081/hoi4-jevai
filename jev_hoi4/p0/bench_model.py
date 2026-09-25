"""P0-1: load open-jev on each device, sanity-check outputs, measure latency."""
import sys, time, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
import torch
from typed_decisions.open_jev import OpenJev

STATE = ("Country: Germany. Date: 1939-08. Ideology: fascism. At war: no. Faction: Axis (leader). "
         "Military factories 58, civilian 42, dockyards 12. Divisions 102 (infantry 70, armor 8, motorized 10). "
         "Neighbors: Poland (fascism? no, non-aligned, 30 divisions, 11 mil factories, guaranteed by France, UK), "
         "France (democratic, 85 divisions), Soviet Union (communist, 180 divisions). Army strength ratio vs Poland 3.1.")
QS = [
    {"type": "choice", "instructions": "Which country is the best target to attack next?", "options": ["Poland", "France", "Soviet Union", "none"]},
    {"type": "choice", "instructions": "What should the army emphasise next?", "options": ["infantry line", "mobile breakthrough", "mountain and jungle", "naval landings", "garrison"]},
    {"type": "noul", "instructions": "This country should prepare for war against Poland within a year."},
    {"type": "noul", "instructions": "This country is weaker than Poland."},
]

def bench(dev, dtype=None, n=5):
    t0 = time.time()
    m = OpenJev.from_pretrained(os.path.join(ROOT), device=dev)
    if dtype is not None:
        m.model.to(dtype)
    load = time.time() - t0
    out = m.decide(STATE, QS)  # warmup
    if dev == "xpu": torch.xpu.synchronize()
    ts = []
    for _ in range(n):
        t = time.time(); out = m.decide(STATE, QS)
        if dev == "xpu": torch.xpu.synchronize()
        ts.append(time.time() - t)
    # batch of 8 identical-shape states
    items = [(STATE, QS)] * 8
    b = m.collator([(s, [m._question(i, q) for i, q in enumerate(qs)]) for s, qs in items], m.device)
    t = time.time()
    with torch.no_grad():
        m.model(b["input_ids"], b["attention_mask"], b["opt_pos"], b["opt_mask"], b["q_pos"], b["seg"])
    if dev == "xpu": torch.xpu.synchronize()
    tb = time.time() - t
    print(f"[{dev} {dtype}] load {load:.1f}s  single {min(ts)*1000:.0f}ms (median {sorted(ts)[len(ts)//2]*1000:.0f})  batch8 {tb*1000:.0f}ms")
    for q, o in zip(QS, out):
        print("   ", q["instructions"][:60], "->", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in o.items() if k != "probabilities"})
    del m

if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count())
    for spec in sys.argv[1:] or ["cpu", "xpu", "xpu:bf16"]:
        dev, _, dt = spec.partition(":")
        bench(dev, {"bf16": torch.bfloat16, "fp16": torch.float16}.get(dt))
