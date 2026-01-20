import pandas as pd
import streamlit as st

CSV_URL = "https://raw.githubusercontent.com/MK316/camp26/refs/heads/main/data/datatotalQ12.csv"

LIKERT_ITEMS = [
    "Q01_GenC","Q02_UndA","Q03_UseA","Q04_SolP","Q05_PedU","Q06_GroB",
    "Q07_IntL","Q08_EmoB","Q09_NeedS","Q10_ValU","Q11_UrgE","Q12_ManI"
]

@st.cache_data(show_spinner=False)
def load_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    for c in ["Field_Group", "Academic_Field", "Year_Level"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    for c in LIKERT_ITEMS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

st.title("Item Distributions (1–6 Likert)")
df = load_data(CSV_URL)

with st.sidebar:
    st.header("Filters")
    fg = st.multiselect("Field_Group", sorted(df["Field_Group"].dropna().unique().tolist()), default=sorted(df["Field_Group"].dropna().unique().tolist()))
    yl = st.multiselect("Year_Level", sorted(df["Year_Level"].dropna().unique().tolist()), default=sorted(df["Year_Level"].dropna().unique().tolist()))
    af = st.multiselect("Academic_Field", sorted(df["Academic_Field"].dropna().unique().tolist()), default=sorted(df["Academic_Field"].dropna().unique().tolist()))
    item = st.selectbox("Select an item", LIKERT_ITEMS, index=0)

fdf = df[
    df["Field_Group"].isin(fg) &
    df["Year_Level"].isin(yl) &
    df["Academic_Field"].isin(af)
].copy()

st.write(f"**Selected item:** {item}")
s = fdf[item].dropna()

if s.empty:
    st.warning("선택한 필터 조건에서 해당 문항 응답이 없습니다.")
    st.stop()

# 1~6 빈도 테이블
counts = s.value_counts().reindex([1,2,3,4,5,6], fill_value=0).reset_index()
counts.columns = ["likert", "count"]
counts["percent"] = (counts["count"] / counts["count"].sum() * 100).round(2)

c1, c2, c3 = st.columns(3)
c1.metric("N", f"{len(s):,}")
c2.metric("Mean", f"{s.mean():.3f}")
c3.metric("SD", f"{s.std(ddof=1):.3f}")

st.subheader("Frequency table")
st.dataframe(counts, use_container_width=True)

st.subheader("Bar chart (counts)")
st.bar_chart(counts.set_index("likert")["count"])

st.subheader("Bar chart (percent)")
st.bar_chart(counts.set_index("likert")["percent"])
