import re, sys
from pathlib import Path
ok = True
for method in ["RBF_NO_ROBUST", "RBF_FROZEN"]:
    p = Path(f"/root/pilot_{method}.log")
    text = p.read_text() if p.exists() else ""
    m = re.search(r"Total episodes:\s*(\d+).*?Total successes:\s*(\d+)", text, re.S)
    if not m:
        print(method, "NO_RESULT"); ok = False; continue
    n, s = int(m.group(1)), int(m.group(2))
    print(method, f"{s}/{n}")
    if s < 8 or n < 10:
        ok = False
print("PILOT_OK" if ok else "PILOT_FAILED")
sys.exit(0 if ok else 1)
