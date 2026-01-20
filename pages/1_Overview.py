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
    # 안전장치: 문자열 컬럼 정리
    for c in ["Field_Group", "Academic_Field", "Year_Level"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # 리커트 문항을 숫자로 강제 (에러는 NaN)
    for c in LIKERT_ITEMS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

st.title("Overview")
df = load_data(CSV_URL)

# 컬럼 존재 확인
missing_cols = [c for c in (META_COLS + LIKERT_ITEMS) if c not in df.columns]
if missing_cols:
    st.error(f"CSV에 다음 컬럼이 없습니다: {missing_cols}")
    st.stop()

st.caption("필터는 다른 페이지에서도 동일한 방식으로 제공됩니다.")

with st.sidebar:
    st.header("Filters")
    fg = st.multiselect("Field_Group", sorted(df["Field_Group"].dropna().unique().tolist()), default=sorted(df["Field_Group"].dropna().unique().tolist()))
    yl = st.multiselect("Year_Level", sorted(df["Year_Level"].dropna().unique().tolist()), default=sorted(df["Year_Level"].dropna().unique().tolist()))
    af = st.multiselect("Academic_Field", sorted(df["Academic_Field"].dropna().unique().tolist()), default=sorted(df["Academic_Field"].dropna().unique().tolist()))

fdf = df[
    df["Field_Group"].isin(fg) &
    df["Year_Level"].isin(yl) &
    df["Academic_Field"].isin(af)
].copy()

col1, col2, col3 = st.columns(3)
col1.metric("Rows (filtered)", f"{len(fdf):,}")
col2.metric("Field_Group (selected)", f"{len(fg):,}")
col3.metric("Academic_Field (selected)", f"{len(af):,}")

st.subheader("Data Preview")
st.dataframe(fdf.head(30), use_container_width=True)

st.subheader("Missingness (Q01–Q12)")
miss = fdf[LIKERT_ITEMS].isna().sum().to_frame("missing_count")
miss["missing_rate"] = (miss["missing_count"] / len(fdf)).round(4)
st.dataframe(miss, use_container_width=True)

st.subheader("Descriptive Stats (Q01–Q12)")
desc = fdf[LIKERT_ITEMS].describe().T
desc = desc.rename(columns={"50%": "median"})
st.dataframe(desc[["count","mean","std","min","median","max"]].round(3), use_container_width=True)
