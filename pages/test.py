import pandas as pd
import streamlit as st
import plotly.express as px

# =========================
# Hum: E-items (Single-choice)
# =========================
st.set_page_config(page_title="Hum E-items (E1–E2)", layout="wide")

# ✅ 실제 CSV raw URL로 바꾸세요
CSV_URL_E = "https://raw.githubusercontent.com/MK316/camp26/refs/heads/main/data/Hum-essay-105.csv"

META_COLS = ["Academic_Field", "Year_Level"]
COL_E1 = "E1"
COL_E2 = "E2"

DISPLAY_LABELS = {
    COL_E1: "[AI 역량의 취업 영향도]",
    COL_E2: "[인문 진로에서 AI 역량 중요성]",
    "Academic_Field": "학문 분야 (Academic_Field)",
    "Year_Level": "학년 (Year_Level)",
}

# ✅ 보기(옵션) 순서 고정: 이미지 그대로
E1_OPTIONS = [
    "전혀 영향을 미치지 않는다.",
    "거의 영향을 미치지 않는다.",
    "보통이다(영향이 크지도 작지도 않다).",
    "어느 정도 영향을 미친다.",
    "매우 큰 영향을 미친다.",
]

E2_OPTIONS = [
    "대부분의 인문 분야에서는 거의 중요하지 않다.",
    "일부 직무에서만 중요하다.",
    "중요도는 중간 정도다(있으면 유리한 수준).",
    "많은 직무에서 점점 필수에 가까워지고 있다.",
    "거의 모든 직무에서 매우 중요하거나 필수라고 본다.",
]

OPTIONS_MAP = {COL_E1: E1_OPTIONS, COL_E2: E2_OPTIONS}


# -------------------------
# Load
# -------------------------
@st.cache_data(show_spinner=False)
def load_data(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(url, encoding="cp949")

    for c in META_COLS + [COL_E1, COL_E2]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df


# -------------------------
# Summary for single-choice
# -------------------------
def single_choice_summary(df: pd.DataFrame, col: str, option_order: list[str]) -> tuple[pd.DataFrame, int]:
    """
    단일선택 문항 요약:
    - option_order에 따라 0 포함
    - percent = count / valid_n * 100
    """
    s = df[col].dropna().astype(str).str.strip()
    s = s[(s != "") & (s.str.lower() != "nan")]

    valid_n = int(len(s))
    if valid_n == 0:
        out0 = pd.DataFrame({"보기": option_order, "빈도": 0, "비율(%)": 0.0})
        return out0, 0

    counts = s.value_counts()
    out = pd.DataFrame({"보기": option_order})
    out["빈도"] = out["보기"].map(counts).fillna(0).astype(int)
    out["비율(%)"] = (out["빈도"] / valid_n * 100).round(2)
    return out, valid_n


def render_single(col: str, fdf: pd.DataFrame, palette_name: str):
    label = DISPLAY_LABELS.get(col, col)
    option_order = OPTIONS_MAP[col]

    st.markdown(f"#### {label}")
    summ, n_valid = single_choice_summary(fdf, col, option_order)
    st.metric("유효 응답 수 (N)", f"{n_valid:,}")

    if n_valid == 0:
        st.info("현재 필터 조건에서 유효 응답이 없습니다.")
        return

    color_seq = getattr(px.colors.qualitative, palette_name, px.colors.qualitative.Plotly)

    # 큰 값이 위로 오게: 가로막대는 ascending으로 두면 위쪽이 큼
    plot_df = summ.sort_values("비율(%)", ascending=True).copy()

    fig = px.bar(
        plot_df,
        x="비율(%)",
        y="보기",
        orientation="h",
        text="비율(%)",
        color="보기",
        color_discrete_sequence=color_seq,
        title=f"{label} 응답 분포(%)"
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
    fig.update_layout(
        height=520,
        showlegend=False,
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis_title="비율(%)",
        yaxis_title=""
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("빈도표")
    st.dataframe(summ, use_container_width=True, hide_index=True)


# =========================
# UI
# =========================
st.markdown("### 🧩 인문 영역: E1–E2 (단일선택) 결과")
st.caption("E1/E2는 단일선택 문항이며, 보기(옵션) 순서는 설문 이미지 기준으로 고정했습니다.")

df = load_data(CSV_URL_E)

required = META_COLS + [COL_E1, COL_E2]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"CSV에 다음 컬럼이 없습니다: {missing}")
    st.stop()

with st.sidebar:
    st.header("필터 (Filters)")

    all_af = sorted(df["Academic_Field"].dropna().astype(str).unique().tolist())
    af = st.multiselect("학문 분야 (Academic_Field)", all_af, default=all_af)

    all_yl = sorted(df["Year_Level"].dropna().astype(str).unique().tolist())
    yl = st.multiselect("학년 (Year_Level)", all_yl, default=all_yl)

    st.divider()
    palette = st.selectbox(
        "색상 팔레트",
        ["Plotly", "D3", "G10", "T10", "Alphabet", "Dark24", "Set2", "Pastel"],
        index=0
    )
    show_raw = st.checkbox("원자료 일부 보기", value=False)

fdf = df[df["Academic_Field"].isin(af) & df["Year_Level"].isin(yl)].copy()

c1, c2, c3 = st.columns(3)
c1.metric("표본 수 (현재 필터 N)", f"{len(fdf):,}")
c2.metric("선택 Academic_Field 수", f"{len(af):,}")
c3.metric("선택 Year_Level 수", f"{len(yl):,}")

if show_raw:
    st.subheader("데이터 미리보기")
    show_df = fdf[META_COLS + [COL_E1, COL_E2]].copy().rename(columns=DISPLAY_LABELS)
    st.dataframe(show_df.head(30), use_container_width=True)

tab1, tab2 = st.tabs([DISPLAY_LABELS[COL_E1], DISPLAY_LABELS[COL_E2]])

with tab1:
    render_single(COL_E1, fdf, palette)

with tab2:
    render_single(COL_E2, fdf, palette)
