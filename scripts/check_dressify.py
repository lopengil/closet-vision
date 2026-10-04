"""Does Dressify tell real outfits from shuffled ones?

Uses the Polyvore Outfits compatibility test (Vasileva et al., ECCV 2018):
real outfits (label 1) vs. outfits with items swapped for same-type items
from other outfits (label 0). Only the needed pictures are read from the
2.5 GB images.zip, via HTTP range requests.

    python scripts/check_dressify.py --n 150 [--split disjoint]

Prints the AUC (0.5 = coin flip, 1.0 = perfect) for Dressify and, as a
baseline, for "items look alike" (mean cosine of fingerprints).
"""
from __future__ import annotations

import argparse
import io
import json
import random
import time
import zipfile

import numpy as np
from huggingface_hub import HfFileSystem, hf_hub_download
from PIL import Image

from closet_vision.style import StyleModel, auc

DATA = "Stylique/Polyvore"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=150, help="outfits per label")
    p.add_argument("--split", default="disjoint")
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()

    outfits = json.load(open(hf_hub_download(DATA, f"{a.split}/test.json", repo_type="dataset")))
    index = {f"{o['set_id']}_{it['index']}": it["item_id"] for o in outfits for it in o["items"]}
    lines = open(hf_hub_download(DATA, f"{a.split}/compatibility_test.txt",
                                 repo_type="dataset")).read().split("\n")
    rows = [l.split() for l in lines if l.strip()]
    rng = random.Random(a.seed)
    pos = rng.sample([r[1:] for r in rows if r[0] == "1"], a.n)
    neg = rng.sample([r[1:] for r in rows if r[0] == "0"], a.n)

    t = time.time()
    zf = zipfile.ZipFile(HfFileSystem().open(f"datasets/{DATA}/images.zip", "rb", block_size=1 << 16))
    names = {n.rsplit("/", 1)[-1].split(".")[0]: n for n in zf.namelist() if n.endswith(".jpg")}
    print(f"zip index: {len(names)} pictures ({time.time() - t:.0f}s)")

    model = StyleModel()
    cache: dict[str, np.ndarray] = {}

    def fingerprints(outfit):
        ids = [index[x] for x in outfit if x in index]
        todo = [i for i in ids if i not in cache and i in names]
        if todo:
            imgs = [Image.open(io.BytesIO(zf.read(names[i]))).convert("RGB") for i in todo]
            for i, e in zip(todo, model.embed(imgs)):
                cache[i] = e
        return [cache[i] for i in ids if i in cache]

    t = time.time()
    P = [fingerprints(o) for o in pos]
    N = [fingerprints(o) for o in neg]
    print(f"{len(cache)} items embedded ({time.time() - t:.0f}s)")

    sp, sn = model.score(P), model.score(N)

    def look_alike(f):
        m = np.asarray(f)
        s = m @ m.T
        k = len(f)
        return float((s.sum() - k) / (k * k - k)) if k > 1 else 0.0

    print(f"Dressify AUC:   {auc(sp, sn):.3f}  (real outfits mean {np.mean(sp):.3f}, "
          f"shuffled {np.mean(sn):.3f})")
    print(f"Look-alike AUC: {auc([look_alike(f) for f in P], [look_alike(f) for f in N]):.3f}")


if __name__ == "__main__":
    main()
