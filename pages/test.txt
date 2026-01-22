import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

CSV_URL = "https://raw.githubusercontent.com/MK316/camp26/refs/heads/main/data/datatotalQ12_0121_120.csv"

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

FIELD_GROUP_LABELS = {
    "ECS": "(IT)공대-컴퓨터 (ECS)",
    "Hum": "인문 (Humanities)",
    "Edu": "사범 (Education)"
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

# -----------------------------
# 3D 느낌(그림자+테두리) 막대 함수
# -----------------------------
def bar_3d_like(x_vals, y_vals, colors, title, xaxis_title, yaxis_title,
               height=520, showlegend=False):
    fig = go.Figure()

    # 막대 그림자(바닥 그림자) - 각 막대 뒤로 살짝 이동된 반투명 사각형
    # (x축이 category여도 shape는 작동하므로, bar 폭은 'paper' 좌표로 처리하지 않고 단순히 느낌만)
    # 카테고리 기반에서는 정확한 폭 제어가 어렵기 때문에 "약한 그림자"로만 연출합니다.
    for i, (x, y) in enumerate(zip(x_vals, y_vals)):
        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=x,
            x1=x,
            y0=0,
            y1=y,
            line=dict(width=0),
            fillcolor="rgba(0,0,0,0.12)",
            layer="below",
        )

    # 실제 막대
    fig.add_trace(
        go.Bar(
            x=x_vals,
            y=y_vals,
            marker=dict(
                color=colors,
                line=dict(color="rgba(0,0,0,0.35)", width=1.2),  # 테두리로 입체감
            ),
            text=[f"{v:.1f}%" if isinstance(v, float) else v for v in y_vals],
            textposition="outside",
            cliponaxis=False,
            opacity=0.95,
            hovertemplate="%{x}<br>%{y}<extra></extra>"
        )
    )

    # 레이아웃: 약간의 깊이감(그리드/배경 최소화)
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=40, r=20, t=60, b=40),
        xaxis=dict(title=xaxis_title, type="category", showgrid=False),
        yaxis=dict(title=yaxis_title, showgrid=True, gridcolor="rgba(0,0,0,0.08)", zeroline=False),
        showlegend=showlegend,
        bargap=0.25,
    )
    return fig

st.markdown("#### 2. 문항별 응답 분포 (Item Distributions)")
st.caption("📌 왼쪽 메뉴에 선택 필터를 조정하세요. (영역별, 항목별, 학과별, 등등 가능)")
df = load_data(CSV_URL)

with st.sidebar:
    st.header("필터 (Filters)")
    with st.expander("필터 펼치기/접기", expanded=False):
        all_fg = sorted(df["Field_Group"].dropna().unique().tolist())
        fg = st.multiselect("Field_Group", all_fg, default=all_fg)

        all_yl = sorted(df["Year_Level"].dropna().unique().tolist())
        yl = st.multiselect("Year_Level", all_yl, default=all_yl)

        all_af = sorted(df["Academic_Field"].dropna().unique().tolist())
        af = st.multiselect("Academic_Field", all_af, default=all_af)

    st.divider()

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

s = fdf[item].dropna()
if s.empty:
    st.warning("선택한 필터 조건에서 해당 문항 응답이 없습니다.")
    st.stop()

counts = likert_counts(fdf[item])

c1, c2, c3, c4 = st.columns(4)
c1.metric("표본 수 (N)", f"{len(s):,}")
c2.metric("평균 (Mean)", f"{s.mean():.3f}")
c3.metric("표준편차 (SD)", f"{s.std(ddof=1):.3f}")
c4.metric("중앙값 (Median)", f"{s.median():.1f}")

tab1, tab2, tab3 = st.tabs([
    "📊 분포 (Distribution)",
    "🧾 요약 (Summary)",
    "👥 그룹 비교 (Group Comparison)"
])

# -----------------------------
# TAB1: 분포 그래프 (3D-like)
# -----------------------------
with tab1:
    st.subheader(f"{ITEM_LABELS[item]}: 응답 분포(%)")

    plot_df = counts.copy()
    plot_df["likert"] = plot_df["likert"].astype(str)

    # 팔레트 옵션 유지(기본은 Plotly)
    palette = st.selectbox(
        "색상 팔레트 선택",
        ["Plotly", "D3", "G10", "T10", "Alphabet", "Dark24", "Set2", "Pastel"],
        index=0,
        help="막대 색상을 바꿉니다."
    )
    color_seq = getattr(px.colors.qualitative, palette, px.colors.qualitative.Plotly)

    x_vals = plot_df["likert"].tolist()
    y_vals = plot_df["percent"].tolist()

    # 1~6 각각 색이 다르게(보고서 느낌)
    colors = [color_seq[i % len(color_seq)] for i in range(len(x_vals))]

    fig_dist = bar_3d_like(
        x_vals=x_vals,
        y_vals=y_vals,
        colors=colors,
        title="리커트 응답 비율(%)",
        xaxis_title="리커트 값 (1–6)",
        yaxis_title="비율(%)",
        height=520
    )

    # 텍스트는 %로
    fig_dist.update_traces(
        text=[f"{v:.1f}%" for v in y_vals],
        hovertemplate="리커트 %{x}<br>%{y:.2f}%<extra></extra>"
    )

    st.plotly_chart(fig_dist, use_container_width=True)
    st.caption("그래프는 각 응답값(1–6)의 비율(%)을 보여줍니다.")

# -----------------------------
# TAB2: 요약 (기존 유지)
# -----------------------------
with tab2:
    st.subheader(f"{ITEM_LABELS[item]}: 핵심 요약")

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

# -----------------------------
# TAB3: 그룹 평균 비교 (3D-like)
# -----------------------------
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

    g["Mean"] = g["Mean"].round(3)
    g["SD"] = g["SD"].round(3)

    # 그룹 순서/라벨
    if group_by == "Field_Group":
        order = ["ECS", "Hum", "Edu"]
        g["Group"] = pd.Categorical(g["Group"], categories=order, ordered=True)
        g = g.sort_values("Group")
        g["Group_Label"] = g["Group"].astype(str).map(FIELD_GROUP_LABELS).fillna(g["Group"].astype(str))
        x_col = "Group_Label"
        xaxis_title = "전공 영역 (Field_Group)"
    elif group_by == "Year_Level":
        g = g.sort_values("Mean", ascending=False)
        x_col = "Group"
        xaxis_title = "학년 (Year_Level)"
    else:
        g = g.sort_values("Mean", ascending=False)
        x_col = "Group"
        xaxis_title = "학과 (Academic_Field)"

    # 팔레트 옵션 유지(탭1에서 이미 선택했으니 같은 값 사용)
    # color_seq 그대로 사용
    x_vals = g[x_col].astype(str).tolist()
    y_vals = g["Mean"].tolist()
    colors = [color_seq[i % len(color_seq)] for i in range(len(x_vals))]

    fig_mean = bar_3d_like(
        x_vals=x_vals,
        y_vals=y_vals,
        colors=colors,
        title=f"{ITEM_LABELS[item]}: 그룹별 평균 (Mean)",
        xaxis_title=xaxis_title,
        yaxis_title="평균 (Mean, 1–6)",
        height=520
    )
    fig_mean.update_traces(
        text=[f"{v:.2f}" for v in y_vals],
        hovertemplate="%{x}<br>Mean=%{y:.3f}<extra></extra>"
    )

    st.plotly_chart(fig_mean, use_container_width=True)
    st.caption("막대는 그룹별 평균(Mean)을 의미합니다. 아래 표에서 표본 수(N)와 표준편차(SD)도 함께 확인하세요.")
    st.dataframe(g, use_container_width=True, hide_index=True)
