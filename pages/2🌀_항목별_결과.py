import pandas as pd
import streamlit as st
import plotly.express as px


CSV_URL = "https://raw.githubusercontent.com/MK316/camp26/refs/heads/main/data/datatotalQ12_01206PM3.csv"

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

st.markdown("#### 2. 문항별 응답 분포 (Item Distributions)")
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

    # likert를 문자열로 만들어서 "1~6 범주형"으로 고정
    plot_df = counts.copy()
    plot_df["likert"] = plot_df["likert"].astype(str)

    fig_dist = px.bar(
        plot_df,
        x="likert",
        y="percent",
        text="percent",
        title="리커트 응답 비율(%)"
    )

    fig_dist.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        cliponaxis=False
    )

    fig_dist.update_layout(
        xaxis_title="리커트 값 (1–6)",
        yaxis_title="비율(%)",
        xaxis=dict(type="category"),   # ✅ 1~6 간격을 확실히 벌림
        bargap=0.25,                   # ✅ 막대 간격
        height=520,                    # ✅ 세로 길이 확보 (가장 중요)
        margin=dict(l=40, r=20, t=60, b=40)
    )

    st.plotly_chart(fig_dist, use_container_width=True)

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

    g = (
        fdf.groupby(group_by)[item]
        .agg(N="count", Mean="mean", SD="std")
        .reset_index()
        .rename(columns={group_by: "Group"})
    )

    if g.empty:
        st.warning("선택한 조건에서 그룹 비교 결과가 없습니다.")
        st.stop()

    # 값 정리
    g["Mean"] = g["Mean"].round(3)
    g["SD"] = g["SD"].round(3)

    # ---------- 그룹 표시 순서 고정(가능한 경우) ----------
    if group_by == "Field_Group":
        # ECS, Edu, Hum 순서 고정
        order = ["ECS", "Hum", "Edu"]
        g["Group"] = pd.Categorical(g["Group"], categories=order, ordered=True)
        g = g.sort_values("Group")

    elif group_by == "Year_Level":
        # 1학년~4학년 + 졸업생 순서 고정 (데이터 표기가 다를 수 있어 최대한 유연하게)
        possible_orders = [
            ["1학년", "2학년", "3학년", "4학년", "졸업생"],
            ["1", "2", "3", "4", "졸업생"],
            ["1st year", "2nd year", "3rd year", "4th year", "graduate"],
        ]
        chosen = None
        for ord_ in possible_orders:
            if set(g["Group"].astype(str)).issubset(set(ord_)) or any(x in ord_ for x in g["Group"].astype(str).unique()):
                chosen = ord_
                break

        if chosen:
            g["Group"] = pd.Categorical(g["Group"].astype(str), categories=chosen, ordered=True)
            g = g.sort_values("Group")
        else:
            # 순서를 못 맞추면 평균 내림차순
            g = g.sort_values("Mean", ascending=False)

    else:
        # Academic_Field 등: 평균 내림차순
        g = g.sort_values("Mean", ascending=False)

    # ---------- 색상 팔레트 선택 ----------
    palette = st.selectbox(
        "색상 팔레트 선택",
        ["Plotly", "D3", "G10", "T10", "Alphabet", "Dark24", "Set2", "Pastel"],
        index=0,
        help="그룹별 막대 색상을 바꿉니다."
    )
    color_seq = getattr(px.colors.qualitative, palette, px.colors.qualitative.Plotly)

    # ---------- Plotly bar chart ----------
    fig = px.bar(
        g,
        x="Group",
        y="Mean",
        color="Group",  # 그룹별 색상
        text="Mean",
        hover_data={"Group": True, "N": True, "Mean": True, "SD": True},
        color_discrete_sequence=color_seq,
        title=f"{ITEM_LABELS[item]}: 그룹별 평균 (Mean)"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
        cliponaxis=False
    )

    fig.update_layout(
        xaxis_title=group_by,
        yaxis_title="Mean (1–6)",
        showlegend=False,  # 범례 필요하면 True
        margin=dict(l=20, r=20, t=70, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption("막대는 그룹별 평균(Mean)을 의미합니다. 아래 표에서 표본 수(N)와 표준편차(SD)도 함께 확인하세요.")
    st.dataframe(g, use_container_width=True, hide_index=True)
