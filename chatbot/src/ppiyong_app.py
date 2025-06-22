from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch
import re, ast, os


import transformers
import peft
import accelerate
import langchain
import chromadb
import pydantic

# FastAPI 앱 설정
app = FastAPI(
    title="AI 응급 처치 챗봇 API",
    description="LoRA 어댑터가 적용된 한국어 LLaMA-3 모델 기반 챗봇입니다.",
    version="1.0.0",
    docs_url="/api/ai/swagger",
    redoc_url="/api/ai/redoc",
    openapi_url="/api/ai/openapi.json",
    swagger_ui_parameters={"syntaxHighlight": False, "defaultModelsExpandDepth": -1}
)

# CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(" 디바이스:", device)

# 모델 정보
base_model_id = "Bllossom/llama-3.2-Korean-Bllossom-3B"
adapter_model_id = "hjjummy/llama-3.2-Korean-emergency-3B-lora-adapter4"
tokenizer_id = "NousResearch/Meta-Llama-3-8B-Instruct"

# Tokenizer 로드
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

# Base 모델 로드
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)

# LoRA 어댑터 결합
os.makedirs("./offload", exist_ok=True)
model = PeftModel.from_pretrained(
    base_model,
    adapter_model_id,
    trust_remote_code=True,
    device_map="auto",
    offload_dir="./offload"
).half().to(device)

# 파이프라인 생성
print(" Pipeline 생성 중...")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# 응답 정제 함수
def clean_emergency_response(text: str) -> str:
    matches = re.findall(r"\{[^{}]+\}", text)
    if not matches:
        return "대처방법:\n응급처치 방법을 인식하지 못했습니다."

    combined_dict = {}

    for block in matches:
        try:
            parsed = ast.literal_eval(block)
            if isinstance(parsed, dict):
                combined_dict.update(parsed)
        except Exception:
            continue  # 무시하고 다음 블록 처리

    if not combined_dict:
        return "대처방법:\n응급처치 정보를 해석하지 못했습니다."

    # key를 숫자 순으로 정렬
    items = sorted(combined_dict.items(), key=lambda x: int(x[0]))
    steps = "\n".join([f"{k}. {v.strip()}" for k, v in items])
    return f"대처방법:\n{steps}"

# 테스트 함수 (로컬 테스트 용)
def test_chat(input_text: str):
    print("\n 질문:", input_text)
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        output = pipe(
            input_text,
            max_new_tokens=300,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
        raw = output[0]["generated_text"]
        print("\n 원본 응답:\n", raw)
        cleaned = clean_emergency_response(raw)
        print("\n 정제된 응답 결과:\n", cleaned)
    except Exception as e:
        print(" 오류:", e)

# 요청 모델
test_chat("화상 응급 처치는 어떻게 해야 하나요?")

# 요청 모델 정의
class Query(BaseModel):
    input: str

@app.post("/api/ai/chat")
async def chat(query: Query):
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        output = pipe(
            query.input,
            max_new_tokens=300,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
        raw = output[0]["generated_text"]
        cleaned = clean_emergency_response(raw)
        return {"response": cleaned}
    except Exception as e:
        return {"error": str(e)}
