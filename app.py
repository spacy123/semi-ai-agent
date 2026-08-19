import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
from fpdf import FPDF
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="공정 라우팅 기반 반도체 AI 에이전트", layout="wide")
st.title("🔬 반도체 장비 장애 및 결과 오류 점검 AI 에이전트 (Query Router 적용)")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_spec_limits():
    spec_path = os.path.join(BASE_DIR, "spec_limits.json")
    if os.path.exists(spec_path):
        try:
            with open(spec_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            st.warning(f"spec_limits.json을 읽는 중 오류가 발생해 기본값을 사용합니다: {e}")
    return {
        "Etch_Depth": {"Target": 150.0, "LSL": 145.0, "USL": 155.0, "Unit": "nm"},
        "CD_Size": {"Target": 30.0, "LSL": 28.0, "USL": 32.0, "Unit": "nm"},
    }

COLUMN_MAPPER = {
    "Etch_Depth_nm": "Etch_Depth",
    "CD_Size_nm": "CD_Size",
    "Oxide_Thickness_nm": "Oxide_Thick",
    "Thickness_nm": "Oxide_Thick",
}

def get_font_path():
    candidates = [
        os.path.join(BASE_DIR, "fonts", "NanumGothic.ttf"),
        "C:/Windows/Fonts/malgun.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

def generate_pdf_report(target_id, status, details, ai_advice):
    pdf = FPDF()
    pdf.add_page()

    font_path = get_font_path()
    font_ok = False
    if font_path:
        try:
            pdf.add_font("KoreanFont", "", font_path, uni=True)
            pdf.set_font("KoreanFont", "", 12)
            font_ok = True
        except Exception as e:
            st.warning(f"한글 폰트 로드 실패: {e}")

    if not font_ok:
        pdf.set_font("Helvetica", "B", 12)

    pdf.cell(0, 10, f"Semiconductor Report - [{target_id}]", ln=True, align="C")
    pdf.ln(5)
    pdf.cell(0, 8, f"Status: {status}", ln=True)
    pdf.ln(3)
    pdf.cell(0, 8, "1. Details:", ln=True)
    for d in details:
        safe_str = str(d).replace("\n", " ")
        pdf.multi_cell(0, 6, f" - {safe_str}")
    pdf.ln(3)
    pdf.cell(0, 8, "2. AI Action Plan:", ln=True)
    pdf.multi_cell(0, 6, str(ai_advice))

    try:
        out_data = pdf.output()
        return bytes(out_data) if isinstance(out_data, (bytes, bytearray)) else out_data.encode("latin-1")
    except Exception as e:
        st.error(f"PDF 생성 중 오류: {e}")
        return b"%PDF-1.4 Mock Output"

tab1, tab2 = st.tabs(["📄 [탭 1] 스마트 라우팅 RAG Q&A", "📊 [탭 2] 데이터 진단 & 종합 분석"])

with tab1:
    st.subheader("🤖 공정 자동 분류(Query Router) 기반 매뉴얼 Q&A")

    manual_dir = os.path.join(BASE_DIR, "manuals")
    if not os.path.exists(manual_dir):
        os.makedirs(manual_dir)

    pdf_files = [f for f in os.listdir(manual_dir) if f.lower().endswith(".pdf")]

    c1, c2 = st.columns([4, 1])
    with c1:
        if pdf_files:
            st.success(f"📚 학습된 매뉴얼 ({len(pdf_files)}개): {', '.join(pdf_files)}")
            st.caption("⚡ 적용 기술: Query Intent Router (토큰 60% 절감), Hybrid Search, EHS Safety Guardrail")
        else:
            st.warning("⚠️ `manuals` 폴더 안에 PDF 매뉴얼 파일이 없습니다.")

    with c2:
        if st.button("🔄 지식베이스 재색인"):
            with st.spinner("스마트 라우팅 지식베이스 재구성 중..."):
                try:
                    from rag_module import create_multi_pdf_rag_chain
                    rag_chain, vectorstore = create_multi_pdf_rag_chain(manual_dir, force_rebuild=True)
                    st.session_state.rag_chain = rag_chain
                    st.toast("✅ 재구성이 완료되었습니다!")
                except Exception as e:
                    st.error(f"재색인 중 오류가 발생했습니다: {e}")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "rag_chain" not in st.session_state and pdf_files:
        with st.spinner("스마트 라우팅 매뉴얼 RAG 엔진 초기화 중..."):
            try:
                from rag_module import create_multi_pdf_rag_chain
                rag_chain, vectorstore = create_multi_pdf_rag_chain(manual_dir)
                st.session_state.rag_chain = rag_chain
            except Exception as e:
                st.error(f"초기화 오류 발생: {e}")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if user_query := st.chat_input("공정 오류, 조치법을 질문하세요 (예: EBR 노즐 정렬 불량 해결법):"):
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            if "rag_chain" in st.session_state:
                with st.spinner("질문 의도 분류 및 타겟 매뉴얼 정밀 검색 중..."):
                    try:
                        history_tuples = [(m["role"], m["content"]) for m in st.session_state.chat_history[:-1]]
                        response = st.session_state.rag_chain.invoke(
                            {"question": user_query, "chat_history": history_tuples}
                        )
                        st.write(response)
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"답변 생성 중 오류가 발생했습니다: {e}")
            else:
                st.write("🤖 매뉴얼 PDF 준비 후 다시 실행하세요.")

with tab2:
    st.subheader("🧪 멀티 CSV 수치 및 장비 로그 정밀 진단")
    specs = load_spec_limits()
    sample_data = pd.DataFrame([
        {"Wafer_ID": "WF-ET01", "Etch_Depth": 151.2, "CD_Size": 30.1, "RF_Power": 752.0},
        {"Wafer_ID": "WF-ET02", "Etch_Depth": 162.5, "CD_Size": 34.8, "RF_Power": 820.0},
    ])

    uploaded_csv = st.file_uploader("분석할 CSV 파일을 업로드하세요", type=["csv"])
    df = sample_data
    if uploaded_csv:
        try:
            df = pd.read_csv(uploaded_csv)
            if df.empty:
                df = sample_data
        except Exception as e:
            df = sample_data

    display_df = df.copy()
    num_cols = display_df.select_dtypes(include=["float64", "int64"]).columns
    display_df[num_cols] = display_df[num_cols].fillna(display_df[num_cols].mean())
    display_df = display_df.fillna("-")

    df_renamed = df.rename(columns=COLUMN_MAPPER)
    st.dataframe(display_df.head(10), use_container_width=True)

    is_log_data = "기록ID" in df.columns or "오류코드" in df.columns or "이벤트유형" in df.columns
    is_spatial_data = "x" in df.columns and "y" in df.columns and len(df) >= 50

    if is_spatial_data:
        st.info("🗺️ **웨이퍼 공간 좌표(Spatial Map) 데이터**가 감지되었습니다.")
        numeric_val_cols = [c for c in df.columns if c not in ["x", "y"] and pd.api.types.is_numeric_dtype(df[c])]
        if numeric_val_cols:
            val_col = numeric_val_cols[0]
            fig = px.density_heatmap(df, x="x", y="y", z=val_col, histfunc="avg", title=f"Wafer Spatial Map - Average {val_col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("x, y 외에 시각화할 수치형 컬럼이 없습니다.")
    else:
        id_col = "기록ID" if "기록ID" in df.columns else ("Wafer_ID" if "Wafer_ID" in df.columns else df.columns[0])
        selected_item = st.selectbox("진단할 ID를 선택하세요:", df[id_col].unique())

        if st.button("🚨 정밀 진단 및 분석 실행"):
            selected_row = df_renamed[df[id_col] == selected_item].iloc[0]

            if is_log_data:
                event_type = str(selected_row.get("이벤트유형", "N/A"))
                err_code = str(selected_row.get("오류코드", "N/A"))
                err_msg = str(selected_row.get("오류명", "N/A"))
                action = str(selected_row.get("조치내용", "N/A"))
                downtime = selected_row.get("다운타임(분)", 0)

                m1, m2, m3 = st.columns(3)
                m1.metric("선택 ID", str(selected_item))
                m2.metric("이벤트 유형", event_type)
                m3.metric("다운타임(분)", f"{downtime} 분")
                st.divider()

                col1, col2 = st.columns([1, 1])
                with col1:
                    if event_type == "오류발생" or (err_code != "N/A" and err_code != "-"):
                        st.error(f"❌ **[{selected_item}] 진단: 장비 오류 기록 ({err_code})**")
                        st.write(f"- **오류명:** {err_msg}")
                        ai_advice = f"조치 가이드: {action}"
                    else:
                        st.success(f"✅ **[{selected_item}] 진단: 정상 점검 기록**")
                        ai_advice = "정기 점검 완료 항목입니다."

                    st.warning(f"💡 **AI 현장 조치 권고사항:**\n{ai_advice}")
                    pdf_bytes = generate_pdf_report(
                        str(selected_item),
                        "ERROR" if event_type == "오류발생" else "PASS",
                        [f"Code: {err_code}", f"Detail: {err_msg}"],
                        ai_advice,
                    )
                    st.download_button("📥 진단 보고서 (PDF) 다운로드", pdf_bytes, file_name=f"Report_{selected_item}.pdf", mime="application/pdf")

                with col2:
                    if "다운타임(분)" in df.columns:
                        top_dt = df.nlargest(6, "다운타임(분)")[[id_col, "다운타임(분)"]]
                        fig = px.bar(top_dt, x=id_col, y="다운타임(분)", color="다운타임(분)", color_continuous_scale="Reds", title="상위 다운타임 발생 기록 (분)")
                        st.plotly_chart(fig, use_container_width=True)
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

                        if pd.isna(val):
                            continue

                        try:
                            val_float = float(val)
                            if val_float == 0.0:
                                continue

                            dev = val_float - target
                            dev_list.append({"Parameter": col_str, "Deviation": dev})

                            if val_float < lsl or val_float > usl:
                                errors.append(f"{col_str}: {val_float}{unit} (스펙: {lsl}~{usl}{unit}, 오차: {dev:+.2f}{unit})")
                        except ValueError:
                            continue

                m1, m2, m3 = st.columns(3)
                m1.metric("선택 ID", str(selected_item))
                m2.metric("진단 상태", "FAIL ❌" if errors else "PASS ✅")
                m3.metric("스펙 이탈 항목 수", f"{len(errors)} 건")
                st.divider()

                col1, col2 = st.columns([1, 1])
                with col1:
                    if errors:
                        st.error(f"❌ **[{selected_item}] 결과: 공정 불량 (FAIL)**")
                        for err in errors:
                            st.write(f"- {err}")
                        ai_advice = "스펙 이탈 감지: 챔버 압력/RF Power를 복구하고 가스 밸브를 점검하세요."
                    else:
                        st.success(f"✅ **[{selected_item}] 결과: 정상 (PASS)**")
                        st.write("- 모든 스펙 항목이 정상 범위 이내입니다.")
                        ai_advice = "정상 공정 수치입니다."

                    st.warning(f"💡 **AI 추천 액션 플랜:**\n{ai_advice}")
                    pdf_bytes = generate_pdf_report(
                        str(selected_item), "FAIL" if errors else "PASS", errors if errors else ["None"], ai_advice
                    )
                    st.download_button("📥 진단 보고서 (PDF) 다운로드", pdf_bytes, file_name=f"Report_{selected_item}.pdf", mime="application/pdf")

                with col2:
                    if dev_list:
                        dev_df = pd.DataFrame(dev_list)
                        fig = px.bar(dev_df, x="Parameter", y="Deviation", color="Deviation", color_continuous_scale="RdBu_r", title=f"{selected_item} 항목별 오차 편차")
                        st.plotly_chart(fig, use_container_width=True)
