import pandas as pd
import streamlit as st

CSV_URL = "https://raw.githubusercontent.com/MK316/camp26/refs/heads/main/data/datatotalQ12.csv"

LIKERT_ITEMS = [
    "Q01_GenC","Q02_UndA","Q03_UseA","Q04_SolP","Q05_PedU","Q06_GroB",
    "Q07_IntL","Q08_EmoB","Q09_NeedS","Q10_ValU","Q11_UrgE","Q12_ManI"
]
META_COLS = ["Field_Group", "Academic_Field", "Year_Level", "Year_Original"]

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

st.title("개요 (Overview)")
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

st.subheader("데이터 미리보기 (Data Preview)")
st.dataframe(fdf.head(30), use_container_width=True)

# 결측 관련: 표 없이 캡션만
st.caption("결측치(Missing)는 없다고 가정하고 분석을 진행합니다.")

# 기술통계 표 (결측 표 자리 대체)
st.subheader("기술통계 (Descriptive Statistics)")
desc = fdf[LIKERT_ITEMS].describe().T
desc = desc.rename(columns={"50%": "median"})
desc_out = desc[["count","mean","std","min","median","max"]].round(3)
desc_out.columns = ["N", "Mean", "SD", "Min", "Median", "Max"]
st.dataframe(desc_out, use_container_width=True)
