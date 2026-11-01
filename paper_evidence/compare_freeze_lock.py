import sys
import subprocess

freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True, encoding="utf-8").splitlines()
freeze_dict = {}
for line in freeze:
    if "==" in line:
        k, v = line.strip().split("==", 1)
        freeze_dict[k.strip().lower().lstrip("\ufeff")] = v.strip()

with open("requirements-lock.txt", encoding="utf-16le") as f:
    content = f.read()
    lock_lines = content.splitlines()

lock_dict = {}
for line in lock_lines:
    line = line.strip().lstrip("\ufeff")
    if line and not line.startswith("#") and "==" in line:
        k, v = line.split("==", 1)
        lock_dict[k.strip().lower()] = v.strip()

mismatches = []
matched = []
missing_in_freeze = []

for k, v in lock_dict.items():
    if k in freeze_dict:
        if freeze_dict[k].lower() == v.lower():
            matched.append((k, v))
        else:
            mismatches.append((k, v, freeze_dict[k]))
    else:
        missing_in_freeze.append((k, v))

print(f"Total lock entries: {len(lock_dict)}")
print(f"Total freeze entries: {len(freeze_dict)}")
print(f"Matched: {len(matched)}")
print(f"Mismatches: {len(mismatches)}")
print(f"Missing in freeze: {len(missing_in_freeze)}")
if mismatches:
    print("Mismatches details:")
    for m in mismatches:
        print(f"  {m[0]}: lock={m[1]} vs freeze={m[2]}")
if missing_in_freeze:
    print("Missing in freeze details:")
    for m in missing_in_freeze:
        print(f"  {m[0]}=={m[1]}")
