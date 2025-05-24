import os
import json
import boto3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableMap, RunnableLambda, RunnableSequence

# ========== 설정 ==========
S3_BUCKET = "ppiyong-s3-bucket"
S3_RAG_PATH = "../../rag_data/knowledge.txt"
S3_CHROMA_PREFIX = "chroma_db/"

LOCAL_RAG_PATH = "/tmp/rag_data/knowledge.txt"
LOCAL_CHROMA_DIR = "/tmp/chroma_db"

base_model_id = "Bllossom/llama-3.2-Korean-Bllossom-3B"
adapter_model_id = "hjjummy/llama-3.2-Korean-emergency-3B-lora-adapter4"

# ========== S3에서 리소스 다운로드 ==========
s3 = boto3.client('s3')

def download_file(bucket, key, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    s3.download_file(bucket, key, dest_path)

def download_chroma_dir(bucket, prefix, local_dir):
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            local_path = os.path.join(local_dir, os.path.relpath(key, prefix))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(bucket, key, local_path)

# 다운로드
download_file(S3_BUCKET, S3_RAG_PATH, LOCAL_RAG_PATH)
download_chroma_dir(S3_BUCKET, S3_CHROMA_PREFIX, LOCAL_CHROMA_DIR)

# ========== 모델 로드 ==========
print("모델 및 토크나이저 로드 중...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

model = PeftModel.from_pretrained(
    base_model,
    adapter_model_id,
    trust_remote_code=True,
    device_map=None
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# ========== 문서 및 검색기 로딩 ==========
documents = TextLoader(LOCAL_RAG_PATH, encoding='utf-8').load()

def split_docs(docs, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

docs = split_docs(documents)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma(persist_directory=LOCAL_CHROMA_DIR, embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})

# ========== 템플릿 및 실행 체인 ==========
instruction_template = """
당신은 응급 상황에서 적절한 대처 방법을 안내하는 응급 처치 전문가입니다.
입력된 상황과 검색된 문서를 분석하여 다음 형식으로 응답하세요:

현재상황 진단: [현재 상황을 간략히 설명]

대처방법:
1. [검색된 문서를 기반으로 한 첫 번째 대처 방법]
2. [검색된 문서를 기반으로 한 두 번째 대처 방법]
3. [검색된 문서를 기반으로 한 세 번째 대처 방법]
"""

prompt = PromptTemplate(template=instruction_template, input_variables=["documents", "input"])

def summarize_documents(doc_list, max_length=500):
    return " ".join([doc.page_content[:max_length] for doc in doc_list])

prompt_runnable = RunnableMap({
    "documents": lambda x: x["documents"],
    "input": lambda x: x["input"],
    "output": prompt
})
pipe_runnable = RunnableLambda(lambda x: pipe(x["output"], max_new_tokens=300, pad_token_id=tokenizer.eos_token_id))
runnable_sequence = RunnableSequence(prompt_runnable, pipe_runnable)

# ========== Lambda 핸들러 ==========
def lambda_handler(event, context):
    try:
        # API Gateway를 통해 들어온 JSON 바디 파싱
        body = json.loads(event.get("body", "{}"))
        user_input = body.get("input")

        if not user_input:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing 'input' in request body"})
            }

        # RAG 흐름 수행
        retrieved_docs = retriever.get_relevant_documents(user_input)
        summarized = summarize_documents(retrieved_docs)
        result = runnable_sequence.invoke({"documents": summarized, "input": user_input})
        answer = result[0]["generated_text"]

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"response": answer})
        }

    except Exception as e:
        print(f"[ERROR] {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "응답 생성 중 오류가 발생했습니다."})
        }