from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.schema.runnable import RunnableSequence, RunnableMap, RunnableLambda
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from peft import PeftModel
import torch

# FastAPI 앱 생성
app = FastAPI(
    title="AI chat bot API",
    description="AI 챗봇 서비스 API입니다.",
    version="1.0.0",
    docs_url="/api/ai/swagger",
    redoc_url="/api/ai/redoc",
    openapi_url="/api/ai/openapi.json",
    swagger_ui_parameters={"syntaxHighlight": False, "defaultModelsExpandDepth": -1}
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://localhost:5173", "http://localhost:3000"],
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["Authorization", "Content-Type"],
    allow_credentials=True,
)

# GPU 확인 및 설정
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA is available. Using {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()
else:
    device = torch.device("cpu")
    print("CUDA is not available. Running on CPU.")

# 모델 및 어댑터 설정
base_model_id = "Bllossom/llama-3.2-Korean-Bllossom-3B"
adapter_model_id = "hjjummy/llama-3.2-Korean-emergency-3B-lora-adapter4"

# config 로드 및 rope_scaling 정제
print("Config 정제 및 모델 로드 중...")
config = AutoConfig.from_pretrained(base_model_id, trust_remote_code=True)
if hasattr(config, "rope_scaling"):
    config.rope_scaling = {
        "type": "linear",
        "factor": config.rope_scaling.get("factor", 32.0)
    }

# 모델 로드
print("모델 로드 중...")
# 모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    config=config,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
# LoRA 어댑터 결합
print("LoRA 어댑터 로드 중...")
model = PeftModel.from_pretrained(
    base_model,
    adapter_model_id,
    trust_remote_code=True,
    device_map="auto"
)
model = model.half().to(device)
# Hugging Face pipeline 생성
print("Pipeline 생성 중...")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0  # CUDA 디바이스 사용
)

# 문서 로드 및 분할
print("문서 로드 및 분할 중...")
text_file_path = "../../rag_data/knowledge.txt"
documents = TextLoader(text_file_path, encoding='utf-8').load()

def split_docs(documents, chunk_size=500, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

docs = split_docs(documents)

# 벡터 스토어 생성
print("벡터 스토어 생성 중...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")

# 검색기 생성
print("Q&A 체인 구성 중...")
retriever = db.as_retriever(search_kwargs={"k": 3})

# 템플릿 정의
instruction_template = """
당신은 응급 상황에서 적절한 대처 방법을 안내하는 응급 처치 전문가입니다.
입력된 상황과 검색된 문서를 분석하여 다음 형식으로 응답하세요:

현재상황 진단: [현재 상황을 간략히 설명]

대처방법:
1. [검색된 문서를 기반으로 한 첫 번째 대처 방법]
2. [검색된 문서를 기반으로 한 두 번째 대처 방법]
3. [검색된 문서를 기반으로 한 세 번째 대처 방법]
...
"""
prompt = PromptTemplate(template=instruction_template, input_variables=["documents", "input"])

# 템플릿 처리 및 파이프라인 구성
prompt_runnable = RunnableMap({
    "documents": lambda x: x["documents"],
    "input": lambda x: x["input"],
    "output": prompt
})
pipe_runnable = RunnableLambda(lambda x: pipe(x["output"]))
runnable_sequence = RunnableSequence(prompt_runnable, pipe_runnable)

# 문서 요약
def summarize_documents(docs, max_length=500):
    summarized = []
    for doc in docs:
        summarized.append(doc[:max_length])
    return " ".join(summarized)

# 요청 모델 정의
class Query(BaseModel):
    input: str

# 질문-응답 API
@app.post("/api/ai/chat")
async def chat(query: Query):
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        retrieved_docs = retriever.get_relevant_documents(query.input)
        summarized_docs = summarize_documents(retrieved_docs, max_length=500)

        formatted_prompt = prompt.format(documents=summarized_docs, input=query.input)

        response = pipe(formatted_prompt,
                        max_new_tokens=300,
                        num_return_sequences=1,
                        pad_token_id=tokenizer.eos_token_id
                        )
        return {"response": response[0]['generated_text']}
    except Exception as e:
        return {"error": str(e)}