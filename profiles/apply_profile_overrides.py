#!/usr/bin/env python3
import json
import shutil
import sys
from pathlib import Path

if len(sys.argv) != 3:
    print("Usage: apply_profile_overrides.py <base_config.json> <overrides.json>")
    sys.exit(1)

base = Path(sys.argv[1]).expanduser().resolve()
over = Path(sys.argv[2]).expanduser().resolve()

if not base.exists():
    print(f"ERROR: base config not found: {base}")
    sys.exit(2)
if not over.exists():
    print(f"ERROR: overrides not found: {over}")
    sys.exit(3)

backup = base.with_suffix(base.suffix + ".bak")
shutil.copy2(base, backup)

with base.open() as f:
    cfg = json.load(f)
with over.open() as f:
    ov = json.load(f)

cfg.update(ov)

with base.open("w") as f:
    json.dump(cfg, f, indent=4)
    f.write("\n")

print(f"Backup: {backup}")
print(f"Applied {len(ov)} keys from: {over}")
print(f"Updated config: {base}")
