FROM runpod/base:0.4.0-cuda11.8.0

# Work dir
WORKDIR /app

# Copy requirements and install
COPY requirements.txt /app/requirements.txt

RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --no-cache-dir -r /app/requirements.txt

# Copy code
COPY rp_handler.py /app/rp_handler.py

# (Optional) set HF cache dir
ENV HF_HOME=/runpod-volume/hf_cache
ENV HUGGINGFACE_HUB_CACHE=/runpod-volume/hf_cache

# If Z-Image ever becomes gated, set HF token at deploy time, e.g.:
# ENV HUGGINGFACE_HUB_TOKEN=your_token_here

# Warmup (optional, can be commented out)
# RUN python3.11 -c "import rp_handler; rp_handler.init_pipeline()"

CMD ["python3.11", "-u", "rp_handler.py"]

