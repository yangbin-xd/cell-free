FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Old pip mis-parses new PyPI metadata (typing_extensions >= 4.16) and the
# pytorch index lacks build deps like flit_core, so upgrade pip and let
# dependencies resolve from PyPI (+cpu pin still comes from the pytorch index).
RUN pip install --no-cache-dir --upgrade pip

# CPU-only PyTorch (avoids 2GB CUDA download)
RUN pip install --no-cache-dir \
    torch==2.4.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple

# torch-geometric
RUN pip install --no-cache-dir torch-geometric==2.6.1

# Remaining dependencies
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.10.1 \
    scikit-learn==1.3.2 \
    matplotlib==3.7.5 \
    plotly==6.6.0 \
    streamlit==1.40.1 \
    "tornado<6.5"

# Copy project files
COPY . .

EXPOSE 7860

CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0"]
