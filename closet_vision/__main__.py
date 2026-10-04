"""closet-vision process photo.jpg [more.jpg ...] --out results/
closet-vision serve [--port 7860]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(prog="closet-vision", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("process", help="cut out + tag photos")
    pr.add_argument("photos", nargs="+")
    pr.add_argument("--out", default="closet-vision-out")
    pr.add_argument("--seg", default=None, help="rembg model, e.g. u2netp")
    pr.add_argument("--no-tagger", action="store_true", help="skip the CLIP model")
    pr.add_argument("--no-hanger-in-photo", action="store_true")
    sv = sub.add_parser("serve", help="run the HTTP API")
    sv.add_argument("--host", default="0.0.0.0")
    sv.add_argument("--port", type=int, default=7860)
    a = p.parse_args(argv)

    if a.cmd == "serve":
        from .server import create_app
        create_app().run(host=a.host, port=a.port, threaded=True)
        return

    from . import cutout, load_tagger, process
    tagger = None if a.no_tagger else load_tagger()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for photo in a.photos:
        r = process(photo, tagger, model=a.seg or cutout.DEFAULT_MODEL,
                    hanger_in_photo=not a.no_hanger_in_photo)
        stem = Path(photo).stem
        r.cutout.save(out / f"{stem}.cutout.png")
        r.hanger.save(out / f"{stem}.hanger.png")
        (out / f"{stem}.json").write_text(json.dumps(r.suggestion.__dict__, indent=2))
        print(f"{photo}: {r.suggestion.name} ({r.suggestion.slot})")


if __name__ == "__main__":
    main()
