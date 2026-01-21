import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go
import numpy as np


CSV_URL = "https://raw.githubusercontent.com/MK316/camp26/refs/heads/main/data/datatotalQ12_0121_120.csv"

LIKERT_ITEMS = [
    "Q01_GenC","Q02_UndA","Q03_UseA","Q04_SolP","Q05_PedU","Q06_GroB",
    "Q07_IntL","Q08_EmoB","Q09_NeedS","Q10_ValU","Q11_UrgE","Q12_ManI"
]
GROUP_COLS = ["Field_Group", "Year_Level", "Academic_Field"]

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
    for c in GROUP_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    for c in LIKERT_ITEMS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def corr_table_by_group(df: pd.DataFrame, x: str, y: str, group_col: str) -> pd.DataFrame:
    rows = []
    for gname, gdf in df.groupby(group_col):
        sub = gdf[[x, y]].dropna()
        n = len(sub)
        if n < 3:
            r = None
        else:
            r = sub[x].corr(sub[y])  # Pearson
        rows.append({"Group": gname, "N": n, "r (Pearson)": r})
    out = pd.DataFrame(rows)
    out["r (Pearson)"] = out["r (Pearson)"].round(3)
    return out.sort_values("N", ascending=False)

st.title("상관 분석 (Correlation Analysis)")
st.caption("두 문항 간 관계를 확인하고, Field_Group별로 패턴이 어떻게 달라지는지 비교합니다.")

df = load_data(CSV_URL)

# ---- Sidebar filters ----
with st.sidebar:
    st.header("필터 (Filters)")

    # 필터는 접어서 공간 확보
    with st.expander("필터 펼치기/접기", expanded=False):
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

    st.divider()

    # 문항 선택 (한글명)
    label_to_code = {v: k for k, v in ITEM_LABELS.items()}
    label_list = [ITEM_LABELS[c] for c in LIKERT_ITEMS]

    x_label = st.selectbox("X축 문항 선택", label_list, index=0)
    y_label = st.selectbox("Y축 문항 선택", label_list, index=1)

    x_item = label_to_code[x_label]
    y_item = label_to_code[y_label]

    group_col = st.selectbox("그룹 기준", ["Field_Group", "Year_Level", "Academic_Field"], index=0)

    # show_trend = st.checkbox("추세선(회귀선) 표시", value=True)
    # show_hist = st.checkbox("전체 문항 상관 히트맵 보기", value=True)

# ---- Filtered data ----
fdf = df[
    df["Field_Group"].isin(fg) &
    df["Year_Level"].isin(yl) &
    df["Academic_Field"].isin(af)
].copy()

# ---- Data for correlation ----
sub = fdf[[group_col, x_item, y_item]].dropna()
if sub.empty:
    st.warning("선택한 필터 조건에서 분석 가능한 데이터가 없습니다.")
    st.stop()

# ---- Tabs ----
tab1, tab2 = st.tabs(["📌 2문항 관계 보기", "🧩 전체 상관 구조(히트맵)"])

with tab1:
    st.subheader(f"{ITEM_LABELS[x_item]} ↔ {ITEM_LABELS[y_item]} 관계")

    c1, c2, c3 = st.columns(3)
    c1.metric("표본 수 (N)", f"{len(sub):,}")
    c2.metric("X 평균", f"{sub[x_item].mean():.2f}")
    c3.metric("Y 평균", f"{sub[y_item].mean():.2f}")

    # 그룹별 상관 테이블
    st.markdown("##### 그룹별 상관계수 (Pearson r)")
    ct = corr_table_by_group(sub, x_item, y_item, group_col)
    st.dataframe(ct, use_container_width=True, hide_index=True)

    # 산점도: 그룹별 색상 + 추세선
    st.markdown("##### 산점도 (그룹별 비교)")
    # 산점도(그룹별 색상)
    fig = px.scatter(
        sub,
        x=x_item,
        y=y_item,
        color=group_col,
        opacity=0.75,
        hover_data={group_col: True, x_item: True, y_item: True},
        labels={
            x_item: ITEM_LABELS[x_item],
            y_item: ITEM_LABELS[y_item],
            group_col: group_col
        },
        title=f"{ITEM_LABELS[x_item]} vs {ITEM_LABELS[y_item]} (색상: {group_col})"
    )
    
    # statsmodels 없이 추세선(전체) 추가
    # statsmodels 없이 추세선(전체) 추가 (항상 표시)
    tmp = sub[[x_item, y_item]].dropna()
    if len(tmp) >= 2:
        x = tmp[x_item].to_numpy()
        y = tmp[y_item].to_numpy()
        a, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 50)
        y_line = a * x_line + b
    
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name="추세선(전체)"
            )
        )
    
    st.plotly_chart(fig, use_container_width=True)
    



with tab2:
    st.subheader("전체 문항 상관 히트맵 (Q01–Q12)")

    corr = fdf[LIKERT_ITEMS].corr().round(3)
    corr_named = corr.rename(index=ITEM_LABELS, columns=ITEM_LABELS)

    fig2 = px.imshow(
        corr_named,
        text_auto=True,
        aspect="auto",
        title="문항 간 상관계수 (Pearson r)"
    )

    # ✅ 텍스트(셀 안 숫자) + 축 글자 크기 2배 정도 확대
    fig2.update_traces(textfont_size=24)  # 기존(기본값) 대비 크게
    fig2.update_layout(
        margin=dict(l=20, r=20, t=60, b=20),
        title_font_size=28,
        xaxis=dict(tickfont=dict(size=20)),
        yaxis=dict(tickfont=dict(size=20))
    )

    st.plotly_chart(fig2, use_container_width=True)
