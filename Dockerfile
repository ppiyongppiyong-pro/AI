# 베이스 이미지: Python 3.10
FROM python:3.10

# 필요 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 코드 복사
COPY ppiyong_app.py ./ppiyong-chatbot

EXPOSE 8000

CMD ["uvicorn", "ppiyong_app.main:app", "--host", "0.0.0.0", "--port", "8000"]