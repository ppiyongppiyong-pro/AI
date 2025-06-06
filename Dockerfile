# Ndivida docker image - cuda 모델 설정
FROM nvidia/cuda:12.5.0-runtime-ubuntu22.04

WORKDIR /ppiyong-chatbot

RUN apt-get update && \
    apt-get install -y python3 python3-pip git && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip install --upgrade pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121 && \
    pip cache purge

COPY . .

RUN pip install --no-cache-dir -r chatbot/src/requirements.txt

EXPOSE 8000

CMD ["uvicorn", "chatbot.src.ppiyong_app:app", "--host", "0.0.0.0", "--port", "8000"]
