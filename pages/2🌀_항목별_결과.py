import pandas as pd
import streamlit as st

CSV_URL = "https://raw.githubusercontent.com/MK316/camp26/refs/heads/main/data/datatotalQ12.csv"

LIKERT_ITEMS = [
    "Q01_GenC","Q02_UndA","Q03_UseA","Q04_SolP","Q05_PedU","Q06_GroB",
    "Q07_IntL","Q08_EmoB","Q09_NeedS","Q10_ValU","Q11_UrgE","Q12_ManI"
]

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

def likert_counts(series: pd.Series) -> pd.DataFrame:
    s = series.dropna()
    counts = s.value_counts().reindex([1, 2, 3, 4, 5, 6], fill_value=0)
    dfc = counts.reset_index()
    dfc.columns = ["likert", "count"]
    total = dfc["count"].sum()
    dfc["percent"] = (dfc["count"] / total * 100).round(2) if total > 0 else 0
    return dfc

st.markdown("#### 문항별 응답 분포 (Item Distributions)")
st.caption("📌 왼쪽 메뉴에 선택 필터를 조정하세요. (영역별, 항목별, 학과별, 등등 가능)")
df = load_data(CSV_URL)

# ---- Sidebar filters ----
# ---- Sidebar filters ----
with st.sidebar:
    st.header("필터 (Filters)")

    # (선택) 필터 전체를 접을 수 있게
    with st.expander("필터 펼치기/접기", expanded=False):

        # --- Field_Group ---
        all_fg = sorted(df["Field_Group"].dropna().unique().tolist())
        fg = st.multiselect(
            "Field_Group",
            all_fg,
            default=all_fg
        )

        # --- Year_Level ---
        all_yl = sorted(df["Year_Level"].dropna().unique().tolist())
        yl = st.multiselect(
            "Year_Level",
            all_yl,
            default=all_yl
        )

        # --- Academic_Field ---
        all_af = sorted(df["Academic_Field"].dropna().unique().tolist())
        af = st.multiselect(
            "Academic_Field",
            all_af,
            default=all_af
        )

    st.divider()

    # 문항 선택/그룹 비교는 expander 밖에 둬서 항상 보이게 (중요!)
    item_label_list = [ITEM_LABELS[c] for c in LIKERT_ITEMS]
    selected_label = st.selectbox("문항 선택", item_label_list, index=0)

    label_to_code = {v: k for k, v in ITEM_LABELS.items()}
    item = label_to_code[selected_label]

    group_by = st.selectbox("그룹 비교 기준", ["Field_Group", "Year_Level", "Academic_Field"], index=0)


# ---- Filtered data ----
fdf = df[
    df["Field_Group"].isin(fg) &
    df["Year_Level"].isin(yl) &
    df["Academic_Field"].isin(af)
].copy()

st.caption("리커트 척도: 1–6 (값이 클수록 해당 진술에 더 동의하는 방향으로 해석 가능)")

# ---- Selected item series ----
s = fdf[item].dropna()
if s.empty:
    st.warning("선택한 필터 조건에서 해당 문항 응답이 없습니다.")
    st.stop()

counts = likert_counts(fdf[item])

# ---- Quick metrics ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("표본 수 (N)", f"{len(s):,}")
c2.metric("평균 (Mean)", f"{s.mean():.3f}")
c3.metric("표준편차 (SD)", f"{s.std(ddof=1):.3f}")
c4.metric("중앙값 (Median)", f"{s.median():.1f}")

# ---- Tabs ----
tab1, tab2, tab3 = st.tabs([
    "📊 분포 (Distribution)",
    "🧾 요약 (Summary)",
    "👥 그룹 비교 (Group Comparison)"
])

with tab1:
    st.subheader(f"{ITEM_LABELS[item]}: 응답 분포(%)")

    # % 막대 그래프용: index=likert, value=percent
    st.bar_chart(counts.set_index("likert")["percent"])

    st.caption("그래프는 각 응답값(1–6)의 비율(%)을 보여줍니다.")

with tab2:
    st.subheader(f"{ITEM_LABELS[item]}: 핵심 요약")

    # 긍정/중립/부정 비율
    total = counts["count"].sum()
    neg = counts.loc[counts["likert"].isin([1, 2]), "count"].sum()
    mid = counts.loc[counts["likert"].isin([3, 4]), "count"].sum()
    pos = counts.loc[counts["likert"].isin([5, 6]), "count"].sum()

    if total > 0:
        neg_p = round(neg / total * 100, 2)
        mid_p = round(mid / total * 100, 2)
        pos_p = round(pos / total * 100, 2)
    else:
        neg_p = mid_p = pos_p = 0.0

    d1, d2, d3 = st.columns(3)
    d1.metric("부정 (1–2) 비율", f"{neg_p:.2f}%")
    d2.metric("중립 (3–4) 비율", f"{mid_p:.2f}%")
    d3.metric("긍정 (5–6) 비율", f"{pos_p:.2f}%")

    st.subheader("응답 분포표")
    st.dataframe(counts, use_container_width=True)

with tab3:
    st.subheader(f"{ITEM_LABELS[item]}: {group_by}별 평균 비교")

    # 그룹별 평균과 N
    g = (
        fdf.groupby(group_by)[item]
        .agg(N="count", Mean="mean", SD="std")
        .reset_index()
    )
    g["Mean"] = g["Mean"].round(3)
    g["SD"] = g["SD"].round(3)

    if g.empty:
        st.warning("선택한 조건에서 그룹 비교 결과가 없습니다.")
    else:
        # 평균 막대 그래프
        chart_df = g.set_index(group_by)["Mean"]
        st.bar_chart(chart_df)

        st.caption("막대는 그룹별 평균(Mean)을 의미합니다. 표에서 표본 수(N)도 함께 확인하세요.")
        st.dataframe(g.rename(columns={group_by: "Group"}), use_container_width=True)
