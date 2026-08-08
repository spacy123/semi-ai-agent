import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
from fpdf import FPDF
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="반도체 공정 이상 진단 에이전트", layout="wide")
st.title("🔬 반도체 장비 장애 및 결과 오류 점검 AI 에이전트")

# --- [기능 1] 공정 스펙 파일 로드 ---
def load_spec_limits():
    if os.path.exists("spec_limits.json"):
        with open("spec_limits.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "Etch_Depth": {"Target": 150.0, "LSL": 145.0, "USL": 155.0, "Unit": "nm"},
        "CD_Size": {"Target": 30.0, "LSL": 28.0, "USL": 32.0, "Unit": "nm"}
    }

# --- [기능 2] PDF 리포트 생성 ---
def generate_pdf_report(target_id, status, details, ai_advice):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Semiconductor Quality Diagnosis Report", ln=True, align="C")
    pdf.ln(8)
    
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, f"Target ID: {target_id}", ln=True)
    pdf.cell(0, 8, f"Status: {status}", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "1. Key Details / Out of Spec Items:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for d in details:
        safe_d = str(d).encode('ascii', 'ignore').decode('ascii')
        pdf.cell(0, 6, f" - {safe_d if safe_d.strip() else 'Detail Record Found'}", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "2. AI Action Plan:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    
    safe_advice = "Inspect RF Power, check pressure valves, and recalibrate sensors." if "FAIL" in status or "오류" in str(details) else "System operating within normal parameters."
    pdf.multi_cell(0, 6, safe_advice)
    
    try:
        out_data = pdf.output()
        if isinstance(out_data, str):
            return out_data.encode('latin-1', 'replace')
        return bytes(out_data)
    except Exception:
        return b"%PDF-1.4 Mock Output"

# --- 탭 구성 ---
tab1, tab2 = st.tabs(["📄 [탭 1] 장비 SOP 매뉴얼 RAG", "📊 [탭 2] 데이터 진단 & 종합 분석"])

# ==========================================
# 탭 1: 대화형 매뉴얼 RAG
# ==========================================
with tab1:
    st.subheader("🤖 통합 반도체 장비 매뉴얼 Q&A (RAG 연동)")
    
    manual_dir = "manuals"
    if not os.path.exists(manual_dir):
        os.makedirs(manual_dir)
        
    pdf_files = [f for f in os.listdir(manual_dir) if f.lower().endswith(".pdf")]
    
    if pdf_files:
        st.success(f"📚 학습된 매뉴얼 파일 ({len(pdf_files)}개): {', '.join(pdf_files)}")
    else:
        st.warning("⚠️ `manuals` 폴더 안에 PDF 매뉴얼 파일이 없습니다.")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    if "rag_chain" not in st.session_state and pdf_files:
        with st.spinner("장비 매뉴얼 지식을 분석 중입니다..."):
            try:
                from rag_module import create_multi_pdf_rag_chain
                rag_chain, retriever, vectorstore = create_multi_pdf_rag_chain(manual_dir)
                st.session_state.rag_chain = rag_chain
                st.toast("✅ 매뉴얼 지식베이스 준비 완료!")
            except Exception as e:
                st.error(f"매뉴얼 학습 중 오류 발생: {e}")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
    if user_query := st.chat_input("장비 조치 절차나 매뉴얼 질문을 입력하세요 (예: PECVD 사용 시 EMO 비상 스위치는 어디에 있어?):"):
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)
            
        with st.chat_message("assistant"):
            if "rag_chain" in st.session_state:
                with st.spinner("매뉴얼 검색 및 AI 답변 생성 중..."):
                    # 대화 히스토리 형식 보완
                    history_tuples = [(m["role"], m["content"]) for m in st.session_state.chat_history[:-1]]
                    response = st.session_state.rag_chain.invoke({
                        "question": user_query,
                        "chat_history": history_tuples
                    })
                    st.write(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
            else:
                fallback_ans = "🤖 `manuals` 폴더에 매뉴얼 PDF를 추가 후 실행하시면 지식 기반 답변이 가능합니다."
                st.write(fallback_ans)
                st.session_state.chat_history.append({"role": "assistant", "content": fallback_ans})

# ==========================================
# 탭 2: 멀티 CSV 대응 정밀 진단 시스템
# ==========================================
with tab2:
    st.subheader("🧪 멀티 CSV 공정 수치 및 장비 장애 데이터 통합 분석")
    
    specs = load_spec_limits()
    
    sample_data = pd.DataFrame([
        {"Wafer_ID": "WF-ET01", "Etch_Depth": 151.2, "CD_Size": 30.1, "RF_Power": 752.0},
        {"Wafer_ID": "WF-ET02", "Etch_Depth": 162.5, "CD_Size": 34.8, "RF_Power": 820.0}
    ])
    
    uploaded_csv = st.file_uploader("공정 수치 또는 장비 이력 CSV 파일을 업로드하세요", type=["csv"])
    
    # 💡 [보완 1] 한글 인코딩 방어 (UTF-8 / CP949 자동 시도)
    if uploaded_csv:
        try:
            df = pd.read_csv(uploaded_csv)
        except UnicodeDecodeError:
            df = pd.read_csv(uploaded_csv, encoding='cp949')
    else:
        df = sample_data
        
    # 💡 [보완 2] 결측치(NaN) 방어
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    df = df.fillna("-")
    
    st.dataframe(df.head(10), use_container_width=True)
    
    is_log_data = "기록ID" in df.columns or "오류코드" in df.columns or "이벤트유형" in df.columns
    id_col = "기록ID" if "기록ID" in df.columns else ("Wafer_ID" if "Wafer_ID" in df.columns else df.columns[0])
    
    # UI 렌더링 동적 컨테이너 처리
    status_box = st.container()
    with status_box:
        if is_log_data:
            st.warning("📋 **장비 유지보수 및 오류 이력 데이터(Log)**가 감지되었습니다.")
        else:
            st.info(f"💡 현재 선택된 수치 데이터 식별 기준 열: **{id_col}**")
        
    selected_item = st.selectbox("진단/조회할 ID(Wafer ID / Time / 기록ID)를 선택하세요:", df[id_col].unique())
    
    if st.button("🚨 정밀 진단 및 분석 실행"):
        selected_row = df[df[id_col] == selected_item].iloc[0]
        
        # --- 케이스 A: 장비 이력 로그 데이터 처리 ---
        if is_log_data:
            event_type = str(selected_row.get("이벤트유형", "N/A"))
            err_code = str(selected_row.get("오류코드", "N/A"))
            err_msg = str(selected_row.get("오류명", "N/A"))
            action = str(selected_row.get("조치내용", "N/A"))
            downtime = selected_row.get("다운타임(분)", 0)
            
            # 💡 [보완 3] KPI 카드
            m1, m2, m3 = st.columns(3)
            m1.metric("선택된 기록 ID", str(selected_item))
            m2.metric("이벤트 유형", event_type)
            m3.metric("다운타임(분)", f"{downtime} 분")
            st.divider()
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if event_type == "오류발생" or (err_code != "N/A" and err_code != "-"):
                    st.error(f"❌ **[{selected_item}] 이력 진단: 장비 오류 기록 ({err_code})**")
                    st.write(f"- **오류명:** {err_msg}")
                    st.write(f"- **상세 설명:** {selected_row.get('오류상세설명', 'N/A')}")
                    ai_advice = f"조치 가이드: {action}"
                else:
                    st.success(f"✅ **[{selected_item}] 이력 진단: 정상 점검/정비 기록**")
                    st.write(f"- **점검 내용:** {selected_row.get('오류상세설명', '정기 점검 수행')}")
                    ai_advice = "정기 점검 완료 항목입니다."
                    
                st.warning(f"💡 **AI 현장 조치 권고사항:**\n{ai_advice}")
                
                details_list = [f"Event: {event_type}", f"Code: {err_code}", f"Downtime: {downtime}m"]
                pdf_bytes = generate_pdf_report(str(selected_item), "ERROR" if event_type=="오류발생" else "PASS", details_list, ai_advice)
                st.download_button("📥 진단 보고서 (PDF) 다운로드", pdf_bytes, file_name=f"Report_{selected_item}.pdf", mime="application/pdf")

            with col2:
                st.write("📊 **장비 다운타임 및 오류 현황 시각화**")
                if "다운타임(분)" in df.columns:
                    top_dt = df.nlargest(6, "다운타임(분)")[[id_col, "다운타임(분)"]]
                    fig = px.bar(top_dt, x=id_col, y="다운타임(분)", color="다운타임(분)", color_continuous_scale="Reds", title="상위 다운타임 발생 기록 (분)")
                    st.plotly_chart(fig, use_container_width=True)

        # --- 케이스 B: 계측 및 수치 센서 데이터 처리 ---
        else:
            errors = []
            dev_list = []
            
            for col, val in selected_row.items():
                col_str = str(col)
                if col_str in specs:
                    target = specs[col_str]["Target"]
                    lsl = specs[col_str]["LSL"]
                    usl = specs[col_str]["USL"]
                    unit = specs[col_str]["Unit"]
                    
                    try:
                        val_float = float(val)
                        dev = val_float - target
                        dev_list.append({"Parameter": col_str, "Deviation": dev})
                        
                        if val_float < lsl or val_float > usl:
                            errors.append(f"{col_str}: {val_float}{unit} (스펙: {lsl}~{usl}{unit}, 오차: {dev:+.2f}{unit})")
                    except ValueError:
                        continue
            
            # 💡 [보완 3] KPI 카드
            m1, m2, m3 = st.columns(3)
            m1.metric("선택된 Wafer/Time ID", str(selected_item))
            m2.metric("진단 상태", "FAIL ❌" if errors else "PASS ✅")
            m3.metric("스펙 이탈 항목 수", f"{len(errors)} 건")
            st.divider()

            col1, col2 = st.columns([1, 1])
            with col1:
                if errors:
                    st.error(f"❌ **[{selected_item}] 결과: 공정 불량 (FAIL)**")
                    for err in errors:
                        st.write(f"- {err}")
                    ai_advice = "스펙 이탈이 감지되었습니다. 챔버 압력/RF Power를 즉시 복구하고 가스 밸브를 점검하세요."
                else:
                    st.success(f"✅ **[{selected_item}] 결과: 정상 (PASS)**")
                    st.write("- 모든 스펙 항목이 정상 허용 범위 이내입니다.")
                    ai_advice = "정상 공정 수치입니다. 추가 조치가 필요하지 않습니다."
                    
                st.warning(f"💡 **AI 추천 액션 플랜:**\n{ai_advice}")
                
                pdf_bytes = generate_pdf_report(str(selected_item), "FAIL" if errors else "PASS", errors if errors else ["None"], ai_advice)
                st.download_button("📥 진단 보고서 (PDF) 다운로드", pdf_bytes, file_name=f"Report_{selected_item}.pdf", mime="application/pdf")
                
            with col2:
                st.write("📊 **Target 수치 대비 편차 시각화 (Feature Importance)**")
                if not dev_list:
                    for c in df.columns:
                        if c != id_col and pd.api.types.is_numeric_dtype(df[c]):
                            try:
                                val_f = float(selected_row[c])
                                mean_val = float(df[c].mean())
                                dev_list.append({"Parameter": str(c), "Deviation": val_f - mean_val})
                            except (ValueError, TypeError):
                                continue
                            if len(dev_list) >= 6:
                                break
                                
                if dev_list:
                    dev_df = pd.DataFrame(dev_list)
                    fig = px.bar(dev_df, x="Parameter", y="Deviation", color="Deviation", color_continuous_scale="RdBu_r", title=f"{selected_item} 항목별 오차 편차")
                    st.plotly_chart(fig, use_container_width=True)