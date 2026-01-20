import pandas as pd
import streamlit as st

CSV_URL = "https://raw.githubusercontent.com/MK316/camp26/refs/heads/main/data/datatotalQ12.csv"

LIKERT_ITEMS = [
    "Q01_GenC","Q02_UndA","Q03_UseA","Q04_SolP","Q05_PedU","Q06_GroB",
    "Q07_IntL","Q08_EmoB","Q09_NeedS","Q10_ValU","Q11_UrgE","Q12_ManI"
]
GROUP_COLS = ["Field_Group", "Year_Level", "Academic_Field"]

@st.cache_data(show_spinner=False)
def load_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    for c in GROUP_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    for c in LIKERT_ITEMS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

st.title("Group Comparison (Mean of items)")
df = load_data(CSV_URL)

with st.sidebar:
    st.header("Filters")
    fg = st.multiselect("Field_Group", sorted(df["Field_Group"].dropna().unique().tolist()), default=sorted(df["Field_Group"].dropna().unique().tolist()))
    yl = st.multiselect("Year_Level", sorted(df["Year_Level"].dropna().unique().tolist()), default=sorted(df["Year_Level"].dropna().unique().tolist()))
    af = st.multiselect("Academic_Field", sorted(df["Academic_Field"].dropna().unique().tolist()), default=sorted(df["Academic_Field"].dropna().unique().tolist()))

    group_by = st.selectbox("Group by", ["Field_Group", "Year_Level", "Academic_Field"], index=0)
    items = st.multiselect("Items", LIKERT_ITEMS, default=LIKERT_ITEMS)

fdf = df[
    df["Field_Group"].isin(fg) &
    df["Year_Level"].isin(yl) &
    df["Academic_Field"].isin(af)
].copy()

if not items:
    st.warning("최소 1개 문항을 선택하세요.")
    st.stop()

# 그룹별 평균
gmean = fdf.groupby(group_by)[items].mean(numeric_only=True).round(3)
gcount = fdf.groupby(group_by)[items].count()

st.subheader("Group means")
st.dataframe(gmean, use_container_width=True)

st.subheader("Group Ns (per item)")
st.dataframe(gcount, use_container_width=True)

st.subheader("Chart (mean by group)")
# st.line_chart / st.bar_chart는 wide-form 가능
st.bar_chart(gmean)
