# Ndivida docker image - cuda 모델 설정
FROM nvidia/cuda:12.1.0-runtime-ubuntu20.04

WORKDIR /ppiyong-chatbot

RUN sed -i 's|http://archive.ubuntu.com/ubuntu|http://mirror.kakao.com/ubuntu|g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y python3 python3-pip git && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip install --upgrade pip

RUN pip install torch==2.2.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121 && \
    pip cache purge

COPY chatbot/src/requirements.txt ./chatbot/src/
RUN pip install --no-cache-dir -r chatbot/src/requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "chatbot.src.ppiyong_app:app", "--host", "0.0.0.0", "--port", "8000"]