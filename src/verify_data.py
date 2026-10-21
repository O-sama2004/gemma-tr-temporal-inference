"""Check the local copy of the dataset against MANIFEST.json.

python src/verify_data.py /content/data
Files listed in the manifest but not uploaded are reported; required files must be present and unchanged.
"""
import hashlib
import json
import os
import sys

REQUIRED = ["splits/train.csv", "splits/test.csv", "splits/test_ids.csv", "structured/structured.csv"]


def main(d):
    man = json.load(open(os.path.join(d, "MANIFEST.json")))
    bad, absent = [], []
    for x in man:
        p = os.path.join(d, x["path"])
        if not os.path.exists(p):
            absent.append(x["path"])
            continue
        if hashlib.sha256(open(p, "rb").read()).hexdigest() != x["sha256"]:
            bad.append(x["path"])
    missing_required = [r for r in REQUIRED if r in absent]
    print(f"checked {len(man) - len(absent)} files | changed: {bad or 'none'} | not uploaded: {len(absent)}")
    if bad or missing_required:
        sys.exit(f"STOP: changed {bad}, missing required {missing_required}")
    print("dataset OK")


if __name__ == "__main__":
    main(sys.argv[1])
