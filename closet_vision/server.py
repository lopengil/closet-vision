"""Tiny HTTP API so a Raspberry Pi (or anything) can use closet-vision remotely.

    closet-vision serve --port 7860
    curl -F photo=@skirt.jpg http://localhost:7860/process

Environment:
    CLOSET_VISION_MODEL   CLIP model id (default patrickjohncyh/fashion-clip)
    CLOSET_VISION_SEG     rembg model (default isnet-general-use; u2netp = fast)
    CLOSET_VISION_TAGGER  set to "off" for cut-out + colours only (no big download)
"""
from __future__ import annotations

import os
import threading

from flask import Flask, jsonify, request

from . import cutout, pipeline

_tagger = None
_lock = threading.Lock()


def get_tagger():
    global _tagger
    if os.environ.get("CLOSET_VISION_TAGGER", "on") == "off":
        return None
    with _lock:
        if _tagger is None:
            from . import load_tagger
            _tagger = load_tagger(os.environ.get("CLOSET_VISION_MODEL"))
    return _tagger


DEMO = """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<title>closet-vision</title>
<style>body{font-family:system-ui;background:#faf7f2;max-width:720px;margin:2rem auto;padding:0 1rem}
img{max-width:45%;background:#fff;border-radius:12px;margin:4px}pre{background:#fff;padding:1rem;border-radius:12px;overflow:auto}</style>
<h1>closet-vision 👗</h1><p>Upload a photo of one clothing item.</p>
<input type=file id=f accept="image/*"><div id=out></div>
<script>
f.onchange=async()=>{out.textContent="Working…";const fd=new FormData();fd.append("photo",f.files[0]);
const r=await (await fetch("process",{method:"POST",body:fd})).json();
out.innerHTML=`<img src="data:image/png;base64,${r.cutout_png}"><img src="data:image/png;base64,${r.hanger_png}">`;
const pre=document.createElement("pre");pre.textContent=JSON.stringify(r.suggestion,null,2);out.append(pre)}
</script>"""


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024

    @app.get("/")
    def demo():
        return DEMO

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.post("/process")
    def process():
        photo = request.files.get("photo")
        if not photo:
            return jsonify(error="send the image as multipart field 'photo'"), 400
        result = pipeline.process(
            photo.stream, get_tagger(),
            model=os.environ.get("CLOSET_VISION_SEG", cutout.DEFAULT_MODEL),
            hanger_in_photo=request.form.get("hanger_in_photo", "1") != "0")
        return jsonify(result.to_json())

    return app
