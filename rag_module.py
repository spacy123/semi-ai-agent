import os
import re
import json
import shutil
import hashlib
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 반도체 주요 약어/용어 표준화 Dictionary
TERM_GLOSSARY = {
    "PR": "Photoresist (감광액)",
    "EBR": "Edge Bead Removal",
    "AF": "Auto Focus",
    "MFC": "Mass Flow Controller",
    "EMO": "Emergency Off",
    "ALD": "Atomic Layer Deposition",
    "ICP": "Inductively Coupled Plasma",
}

KOREAN_KEYWORDS = {
    "photo": ["노광", "감광액"],
    "etch": ["식각"],
    "deposition": ["증착", "박막"],
}
ENGLISH_KEYWORDS = {
    "photo": ["pr", "ebr", "mask", "coater", "focus", "aligner", "photo", "phe", "reticle", "overlay"],
    "etch": ["etch", "icp", "rie", "bosch", "scallop"],
    "deposition": ["cvd", "ald", "pecvd", "precursor", "mfc"],
}

def rewrite_query_with_glossary(query: str) -> str:
    expanded_query = query
    for abbr, full_term in TERM_GLOSSARY.items():
        if re.search(rf"\b{re.escape(abbr)}\b", query, flags=re.IGNORECASE) and full_term not in query:
            expanded_query += f" ({full_term})"
    return expanded_query

def route_question(query: str) -> str:
    q = query.lower()
    for category, words in KOREAN_KEYWORDS.items():
        if any(w in query for w in words):
            return category
    for category, words in ENGLISH_KEYWORDS.items():
        for w in words:
            if re.search(rf"\b{re.escape(w)}\b", q):
                return category
    return "all"

def format_docs(docs):
    if not docs:
        return "검색된 관련 매뉴얼 내용이 없습니다."
    formatted = []
    for doc in docs:
        source_file = os.path.basename(doc.metadata.get("source", "Unknown Manual"))
        page_num = doc.metadata.get("page", 0) + 1
        formatted.append(f"[참고 문서: {source_file} (p.{page_num})]\n{doc.page_content}")
    return "\n\n".join(formatted)

def _compute_manifest(manual_dir: str, pdf_files: list) -> str:
    parts = []
    for pdf in sorted(pdf_files):
        path = os.path.join(manual_dir, pdf)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0
        parts.append(f"{pdf}:{mtime}")
    joined = "|".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()

def _load_stored_manifest(index_path: str):
    manifest_file = os.path.join(index_path, "manifest.json")
    if os.path.exists(manifest_file):
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                return json.load(f).get("manifest_hash")
        except (json.JSONDecodeError, OSError):
            return None
    return None

def _save_manifest(index_path: str, manifest_hash: str):
    os.makedirs(index_path, exist_ok=True)
    manifest_file = os.path.join(index_path, "manifest.json")
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump({"manifest_hash": manifest_hash}, f)

def create_multi_pdf_rag_chain(manual_dir="manuals", force_rebuild=False):
    if not os.path.exists(manual_dir):
        os.makedirs(manual_dir)

    pdf_files = [f for f in os.listdir(manual_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise FileNotFoundError(f"'{manual_dir}' 폴더 안에 PDF 매뉴얼 파일이 없습니다.")

    embeddings = OpenAIEmbeddings()
    index_path = "faiss_index"

    current_manifest = _compute_manifest(manual_dir, pdf_files)
    stored_manifest = _load_stored_manifest(index_path)
    manifest_changed = current_manifest != stored_manifest

    if (force_rebuild or manifest_changed) and os.path.exists(index_path):
        shutil.rmtree(index_path)

    should_rebuild_index = force_rebuild or manifest_changed or not os.path.exists(index_path)

    all_docs = []
    for pdf in pdf_files:
        pdf_path = os.path.join(manual_dir, pdf)
        category = "photo" if "photo" in pdf.lower() else ("etch" if "etch" in pdf.lower() else "deposition")
        try:
            loader = PyMuPDFLoader(pdf_path)
            docs = loader.load()
            for d in docs:
                d.metadata["category"] = category
            all_docs.extend(docs)
        except Exception as e:
            print(f"파일 로딩 실패 [{pdf}]: {e}")

    if not all_docs:
        raise ValueError("PDF 매뉴얼에서 텍스트를 추출할 수 없습니다.")

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)

    parent_docs = parent_splitter.split_documents(all_docs)
    child_docs = child_splitter.split_documents(parent_docs)

    if should_rebuild_index:
        vectorstore = FAISS.from_documents(documents=child_docs, embedding=embeddings)
        vectorstore.save_local(index_path)
        _save_manifest(index_path, current_manifest)
    else:
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    categories = ["photo", "etch", "deposition"]
    bm25_by_category = {}
    for category in categories:
        target_docs = [d for d in child_docs if d.metadata.get("category") == category]
        if target_docs:
            bm25 = BM25Retriever.from_documents(target_docs)
            bm25.k = 3
            bm25_by_category[category] = bm25

    bm25_all = BM25Retriever.from_documents(child_docs)
    bm25_all.k = 3

    template = """당신은 반도체 팹(Fab) 라인 설비 장애 지능형 대응 에이전트입니다.
제공된 [설비 매뉴얼 Context]만을 기반으로 정확하게 답변하세요.

[EHS 및 답변 규칙]
1. 답변 맨 첫 줄에는 안전 주의사항을 출력하세요.
   - Context에 위험 요소가 명시되어 있다면: 🚨 [안전 주의사항 (EHS)]: (관련 위험 요소)
   - 없다면: 🚨 [안전 주의사항 (EHS)]: 장비 가동 전 가스 누출 및 메인 전원 상태를 점검하십시오.
2. 장애 해결 조치 절차는 1, 2, 3 단계별 번호를 붙여 요약하세요.
3. [검증 가드레일]: 질문에 대한 답변이나 조치법이 [설비 매뉴얼 Context]에 전혀 없다면 아래 고정 문구만 출력하세요:
   "제공된 매뉴얼에서 해당 내용을 찾을 수 없습니다."
4. 답변 맨 마지막 줄에는 참고한 문서명과 페이지 번호를 명시하세요:
   📌 [출처 매뉴얼]: 문서명 p.N

[설비 매뉴얼 Context]:
{context}

Answer in Korean:"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0, streaming=True)

    def get_context(dict_input):
        raw_query = dict_input["question"]
        category = route_question(raw_query)
        expanded_query = rewrite_query_with_glossary(raw_query)

        if category != "all" and category in bm25_by_category:
            bm25_retriever = bm25_by_category[category]
            filter_dict = {"category": category}
        else:
            bm25_retriever = bm25_all
            filter_dict = None

        faiss_retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3, "filter": filter_dict} if filter_dict else {"k": 3}
        )

        hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5],
        )

        docs = hybrid_retriever.invoke(expanded_query)
        return format_docs(docs)

    rag_chain = (
        {
            "context": get_context,
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, vectorstore
