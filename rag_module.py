import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# [1] 문서 포맷팅 함수 (파일명 및 페이지 정확히 명시)
def format_docs(docs):
    if not docs:
        return "검색된 관련 매뉴얼 내용이 없습니다."
    
    formatted = []
    for doc in docs:
        source_file = os.path.basename(doc.metadata.get("source", "Unknown Manual"))
        page_num = doc.metadata.get("page", 0) + 1
        formatted.append(f"[참고 문서: {source_file} (p.{page_num})]\n{doc.page_content}")
    return "\n\n".join(formatted)

# [2] manuals 폴더 내의 모든 PDF를 읽어 통합 RAG 체인을 생성하는 함수 (속도 개선 캐싱 반영)
def create_multi_pdf_rag_chain(manual_dir="manuals"):
    if not os.path.exists(manual_dir):
        os.makedirs(manual_dir)
        
    pdf_files = [f for f in os.listdir(manual_dir) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        raise FileNotFoundError(f"'{manual_dir}' 폴더 안에 PDF 매뉴얼 파일이 없습니다.")

    embeddings = OpenAIEmbeddings()
    index_path = "faiss_index"

    # 로컬에 저장된 DB가 있으면 0.1초 만에 바로 로드 (속도 극대화)
    if os.path.exists(index_path):
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        all_docs = []
        for pdf in pdf_files:
            pdf_path = os.path.join(manual_dir, pdf)
            try:
                loader = PyMuPDFLoader(pdf_path)
                docs = loader.load()
                all_docs.extend(docs)
            except Exception as e:
                print(f"파일 로딩 실패 [{pdf}]: {e}")

        if not all_docs:
            raise ValueError("PDF 매뉴얼에서 텍스트를 추출할 수 없습니다.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=150,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        split_documents = text_splitter.split_documents(all_docs)

        vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
        vectorstore.save_local(index_path)  # 캐시 저장

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    template = """당신은 반도체 팹(Fab) 라인 설비 장애 지능형 대응 에이전트입니다.
이전 대화 맥락과 제공된 [설비 매뉴얼 Context]를 함께 참고하여 현장 엔지니어의 질문에 정확히 답변하세요.

[EHS 및 답변 규칙]
1. 답변 맨 첫 줄에는 안전 주의사항을 출력하세요. 
   - Context에 관련 안전 수칙이나 위험성이 명시되어 있다면: 🚨 [안전 주의사항 (EHS)]: (관련 위험 요소 및 점검 필수 항목)
   - Context에 안전 관련 언급이 특별히 없다면: 🚨 [안전 주의사항 (EHS)]: 장비 가동 전 가스 누출 및 메인 전원 상태를 점검하십시오.
2. 장애 해결 조치 절차는 1, 2, 3 번호를 붙여 현장에서 따라 하기 쉽게 단계별로 요약하세요.
3. 질문에 대한 답변이나 조치법이 [설비 매뉴얼 Context]에 전혀 없다면, 추측하지 말고 반드시 아래 고정 문구만을 출력하세요:
   "제공된 매뉴얼에서 해당 내용을 찾을 수 없습니다."
4. 답변 맨 마지막 줄에는 참고한 문서명과 페이지 번호를 명시하세요:
   📌 [출처 매뉴얼]: 문서명 p.N (예: 📌 [출처 매뉴얼]: RIE_operation_manual.pdf p.12)

[설비 매뉴얼 Context]:
{context}

Answer in Korean:"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0, streaming=True)

    rag_chain = (
        {
            "context": (lambda x: x["question"]) | retriever | format_docs,
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever, vectorstore