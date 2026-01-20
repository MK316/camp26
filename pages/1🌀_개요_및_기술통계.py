import pandas as pd
import streamlit as st

CSV_URL = "https://raw.githubusercontent.com/MK316/camp26/refs/heads/main/data/datatotalQ12.csv"

# 원본 리커트 문항 컬럼명
LIKERT_ITEMS = [
    "Q01_GenC","Q02_UndA","Q03_UseA","Q04_SolP","Q05_PedU","Q06_GroB",
    "Q07_IntL","Q08_EmoB","Q09_NeedS","Q10_ValU","Q11_UrgE","Q12_ManI"
]
META_COLS = ["Field_Group", "Academic_Field", "Year_Level", "Year_Original"]

# 화면 표시용 문항 이름 매핑
ITEM_LABELS = {
    "Q01_GenC": "Q01 전반적 인식",
    "Q02_UndA": "Q02 이해 능력",
    "Q03_UseA": "Q03 활용 능력",
    "Q04_SolP": "Q04 문제 해결",
    "Q05_PedU": "Q05 교육적 활용",
    "Q06_GroB": "Q06 성장 인식",
    "Q07_IntL": "Q07 학습 의향",
    "Q08_EmoB": "Q08 정서적 부담감",
    "Q09_NeedS": "Q09 지원 필요",
    "Q10_ValU": "Q10 가치 인식",
    "Q11_UrgE": "Q11 AI 교육의 시급성",
    "Q12_ManI": "Q12 AI 교육의 제도화",
}

@st.cache_data(show_spinner=False)
def load_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)

    # 문자열 컬럼 정리
    for c in ["Field_Group", "Academic_Field", "Year_Level"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # 리커트 문항을 숫자로 강제 (에러는 NaN)
    for c in LIKERT_ITEMS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

st.markdown("#### 1. 설문 결과 개요 (Overview)")
df = load_data(CSV_URL)

# 컬럼 존재 확인
missing_cols = [c for c in (META_COLS + LIKERT_ITEMS) if c not in df.columns]
if missing_cols:
    st.error(f"CSV에 다음 컬럼이 없습니다: {missing_cols}")
    st.stop()

with st.sidebar:
    st.header("필터 (Filters)")
    fg = st.multiselect(
        "Field_Group",
        sorted(df["Field_Group"].dropna().unique().tolist()),
        default=sorted(df["Field_Group"].dropna().unique().tolist())
    )
    yl = st.multiselect(
        "Year_Level",
        sorted(df["Year_Level"].dropna().unique().tolist()),
        default=sorted(df["Year_Level"].dropna().unique().tolist())
    )
    af = st.multiselect(
        "Academic_Field",
        sorted(df["Academic_Field"].dropna().unique().tolist()),
        default=sorted(df["Academic_Field"].dropna().unique().tolist())
    )

fdf = df[
    df["Field_Group"].isin(fg) &
    df["Year_Level"].isin(yl) &
    df["Academic_Field"].isin(af)
].copy()

col1, col2, col3 = st.columns(3)
col1.metric("표본 수 (N)", f"{len(fdf):,}")
col2.metric("선택 Field_Group", f"{len(fg):,}")
col3.metric("선택 Academic_Field", f"{len(af):,}")

st.caption("결측치(Missing)는 없음.")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs([
    "🌱 문항 안내 (Item Guide)",
    "🌱 데이터 미리보기 (Data Preview)",
    "🌱 기술통계 (Descriptive Statistics)"
])

with tab1:
    st.subheader("공통 문항 12개")
    guide_df = pd.DataFrame({
        "코드 (Code)": LIKERT_ITEMS,
        "문항명 (Korean Label)": [ITEM_LABELS[c] for c in LIKERT_ITEMS]
    })
    st.dataframe(guide_df, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("필터 적용 후 상위 30행")
    preview_cols = META_COLS + LIKERT_ITEMS
    preview_df = fdf[preview_cols].rename(columns=ITEM_LABELS)
    st.dataframe(preview_df.head(30), use_container_width=True)

with tab3:
    st.subheader("문항별 기술통계")
    desc = fdf[LIKERT_ITEMS].describe().T
    desc = desc.rename(columns={"50%": "median"})
    desc_out = desc[["count","mean","std","min","median","max"]].round(3)
    desc_out.columns = ["N", "Mean", "SD", "Min", "Median", "Max"]
    desc_out = desc_out.rename(index=ITEM_LABELS)
    st.dataframe(desc_out, use_container_width=True)
