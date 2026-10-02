"""Compare CPU prefill throughput of the VLM at bfloat16 vs float32."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
torch.set_num_threads(12)
from qwen_drive import QwenDriveForPlanning

print("bench start", flush=True)
for name, dt in (("bfloat16", torch.bfloat16), ("float32", torch.float32)):
    t = time.time()
    m = QwenDriveForPlanning.from_pretrained(
        "weights/Qwen-Drive-1.0-4B", dtype=dt, attn_implementation="sdpa").eval()
    vlm = m.vlm
    print(f"[{name}] load {time.time()-t:.1f}s", flush=True)
    for n in (256, 1024):
        ids = torch.randint(0, 1000, (1, n))
        mm = torch.zeros_like(ids)
        t = time.time()
        with torch.no_grad():
            out = vlm(input_ids=ids, mm_token_type_ids=mm, use_cache=True)
        d = time.time() - t
        print(f"[{name}] prefill {n:5d} tok: {d:7.2f}s  ({n/d:8.1f} tok/s)", flush=True)
    # one autoregressive decode step on top of the 1024-token cache
    t = time.time()
    with torch.no_grad():
        vlm(input_ids=torch.tensor([[5]]), mm_token_type_ids=torch.tensor([[0]]),
            past_key_values=out.past_key_values, use_cache=True)
    print(f"[{name}] decode 1 tok:      {time.time()-t:7.2f}s", flush=True)
    del m, vlm, out
    import gc; gc.collect()
print("bench done", flush=True)
