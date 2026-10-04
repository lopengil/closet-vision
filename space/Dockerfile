# Hugging Face Docker Space for closet-vision (also runs anywhere Docker does)
FROM python:3.11-slim
RUN useradd -m user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    U2NET_HOME=/home/user/.u2net HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app
RUN pip install --no-cache-dir --user torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --user "closet-vision[tagger,server,style] @ git+https://github.com/lopengil/closet-vision.git"
# warm the model caches at build time so the first request is fast
RUN python -c "from closet_vision import cutout, load_tagger; cutout._session('isnet-general-use'); load_tagger()" && \
    python -c "from closet_vision.style import StyleModel; StyleModel()"
EXPOSE 7860
CMD ["closet-vision", "serve", "--port", "7860"]
