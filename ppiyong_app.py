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
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

from peft import PeftModel
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
import torch

# FastAPI 앱 생성
app = FastAPI(
    title="AI chat bot API",
    description="AI 챗봇 서비스 API입니다.",
    version="1.0.0",
    docs_url="/api/ai/swagger",
    redoc_url="/api/ai/redoc",
    openapi_url="/api/ai/openapi.json",
    swagger_ui_parameters={"syntaxHighlight":False, "defaultModelsExpandDepth":-1} #하이라이트 비활성화
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://13.125.7.215:8000", "http://13.125.7.215:3000", "http://localhost:3000"],
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["Authorization", "Content-Type"],
    allow_credentials=True,
)


if torch.cuda.is_available():
    torch.cuda.ipc_collect()  # GPU 메모리 공유
else:
    print("CUDA is not available. Running on CPU.")


# Step 1: 원본 모델과 LoRA 어댑터 설정
base_model_id = "Bllossom/llama-3.2-Korean-Bllossom-3B"  # 원본 모델 ID
adapter_model_id = "hjjummy/llama-3.2-Korean-emergency-3B-lora-adapter4"  # LoRA 어댑터 ID

# Step 2: 원본 모델과 토크나이저 로드 (메모리 최적화 포함)
print("원본 모델 로드 중...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    trust_remote_code=True,
    device_map="cpu",               # GPU 활용
    torch_dtype=torch.float16,
    #offload_folder="./offload", # 메모리 절약
    #offload_state_dict=True,
    low_cpu_mem_usage=True           # 메모리 최적화 옵션
).to(torch.device("cpu"))
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# Step 3-1: LoRA 어댑터 결합
print("LoRA 어댑터 로드 중...")
model = PeftModel.from_pretrained(base_model,
                                  adapter_model_id,
                                  trust_remote_code=True,
                                  device_map=None).to(torch.device("cpu"))
print(model.config)


# Step 3-2: 모델을 Hugging Face Pipeline으로 변환
print("Hugging Face Pipeline 생성 중...")
#pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)  # GPU 사용
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)  # GPU 사용  ## device=0 오류나서 수정함.

# Step 4: 텍스트 파일 로드 및 문서 분할
print("문서 로드 및 분할 중...")
text_file_path = "./rag_data/knowledge.txt" # 텍스트 파일 경로
documents = TextLoader(text_file_path, encoding='utf-8').load()

#텍스트분할
def split_docs(documents, chunk_size=500, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

docs = split_docs(documents)

# Step 5: 벡터 스토어 생성
print("벡터 스토어 생성 중...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")

# Step 6: Q&A 체인 구성(검색기 생성)
print("Q&A 체인 구성 중...")

retriever = db.as_retriever(search_kwargs={"k": 3})  # 상위 3개의 문서만 검색

# 응답 형식 템플릿
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
# PromptTemplate 생성
prompt = PromptTemplate(template=instruction_template, input_variables=["documents", "input"])

# RunnableMap을 사용하여 템플릿 실행 가능 객체 생성
prompt_runnable = RunnableMap({
    "documents": lambda x: x["documents"],
    "input": lambda x: x["input"],
    "output": prompt  # 템플릿 생성
})
# RunnableLambda를 사용하여 pipe를 실행 가능 객체로 변환
pipe_runnable = RunnableLambda(lambda x: pipe(x["output"]))

# RunnableSequence를 사용하여 템플릿과 모델 연결
runnable_sequence = RunnableSequence(
    prompt_runnable,  # 템플릿 실행
    pipe_runnable  # 모델 실행
)

# Step 7: 문서 요약 함수
def summarize_documents(docs, max_length=500):
    summarized = []
    for doc in docs:
        summarized.append(doc[:max_length])  # 각 문서를 최대 max_length로 제한
    return " ".join(summarized)

class Query(BaseModel):
    input: str


# # Step 8: 질문-답변 API 생성
# @app.post("/ask")
# async def ask_question(query: Query):
#     try:
#
#         torch.cuda.empty_cache()  # GPU 캐시 메모리 해제
#         torch.cuda.ipc_collect()  # 공유 메모리 수집
#
#         # 검색된 문서 가져오기 및 요약
#         retrieved_docs = retriever.get_relevant_documents(query.input)
#
#         summarized_docs = summarize_documents(retrieved_docs, max_length=500)
#
#         # 템플릿 생성 및 포맷팅
#         formatted_prompt = prompt.format(documents=summarized_docs, input=query.input)  # 이미 문자열로 반환됨
#
#         # RAG를 사용하여 답변 생성
#         response = pipe(formatted_prompt,
#                         max_new_tokens=300,
#                         num_return_sequences=1,
#                         use_cache=True,
#                         pad_token_id=tokenizer.eos_token_id
#                         )
#         return {"response": response[0]['generated_text']}
#     except Exception as e:
#         return {"error": str(e)}


# Step 8-2: 질문-답변 API 생성
# FastAPI 엔드포인트 정의
@app.post("/api/ai/chat")
async def chat(query: Query):
    try:
        # 모델을 사용해 응답 생성
        response = pipe(query.input,
                        max_new_tokens=300,
                        num_return_sequences=1,
                        pad_token_id=tokenizer.eos_token_id
                        )
        return {"response": response[0]['generated_text']}
    except Exception as e:
        return {"error": str(e)}