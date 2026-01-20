import re
import pandas as pd
import streamlit as st
import plotly.express as px

# =========================
# ECS: B-items (Multi-select + Open-ended)
# =========================
st.set_page_config(page_title="ECS B-items", layout="wide")

# ✅ 여기에 실제 CSV raw URL을 넣으세요
CSV_URL_B = "https://raw.githubusercontent.com/MK316/camp26/refs/heads/main/data/IT-essay-116.csv"

# ✅ 실제 컬럼명(사용자 제공 이미지 기준)
META_COLS = ["Field_Group", "Year_Level"]

COL_B1 = "B1. [교육과정·정책 개선 요구]"
COL_B1_OTHER = "B1-기타"

COL_B2 = "B2. [정서적 부담감의 원인]"
COL_B2_OTHER = "B2-기타"

COL_B3 = "B3. [학습 내용 선호]"
COL_B3_OTHER = "B3-기타"       # CSV가 B3-기타면 이걸 쓰세요
COL_B3_OTHER2 = "B2-기타2"     # CSV가 B2-기타2로 되어 있으면 이걸 쓰세요 (둘 중 존재하는 것 자동 선택)

COL_B4 = "B4. [대학에 요구사항]"

B_MULTI = [COL_B1, COL_B2, COL_B3]
B_OPEN = COL_B4

# ✅ 화면에 표시할 때는 [ ] 키워드 중심으로만
B_KEYWORDS = {
    COL_B1: "[교육과정·정책 개선 요구]",
    COL_B2: "[정서적 부담감의 원인]",
    COL_B3: "[학습 내용 선호]",
    COL_B4: "[대학에 요구사항]",
}

# =========================
# ✅ 설문 옵션 목록(고정) - 순서 유지
# =========================
B1_OPTIONS = [
    "기초 수준의 단계별 교육 제공",
    "실습 중심의 수업 확대",
    "전공 및 교과와 연계된 활용 사례 제시",
    "평가 부담이 낮은 수업 설계",
    "정규 교과 내 필수 또는 선택 과목 개설",
    "기타",
]

B2_OPTIONS = [
    "기초 지식이나 사전 경험의 부족",
    "전문 용어나 기술적 내용(예: 코딩)에 대한 두려움",
    "학습 속도를 따라가기 어렵다는 우려",
    "일자리와 직접 관련되지 않는 수업이라는 우려",
    "디지털·AI 학습의 필요성에 대한 확신 부족",
    "기타",
]

B3_OPTIONS = [
    "디지털·AI의 기초 개념과 작동 원리",
    "코딩 및 앱 개발",
    "AI 도구 활용 방법",
    "수업 설계 및 교육적 활용 방법",
    "AI 윤리 및 책임 있는 활용",
    "현장 엔지니어(공학자)와의 교류",
    "기타",
]

OPTIONS_MAP = {
    COL_B1: B1_OPTIONS,
    COL_B2: B2_OPTIONS,
    COL_B3: B3_OPTIONS,
}

# =========================
# Load & helpers
# =========================
@st.cache_data(show_spinner=False)
def load_data(url: str) -> pd.DataFrame:
    # ✅ 한글 인코딩 안전하게 읽기
    try:
        df = pd.read_csv(url, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(url, encoding="cp949")

    # 문자열 컬럼 정리
    for c in META_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # 모든 B문항은 문자열로 정리
    for c in [COL_B1, COL_B1_OTHER, COL_B2, COL_B2_OTHER, COL_B3, COL_B3_OTHER, COL_B3_OTHER2, COL_B4]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    return df


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s).strip()

def is_no_response(text: str) -> bool:
    t = clean_text(str(text)).lower()
    if t in {"", "nan", "none"}:
        return True
    t = re.sub(r"\s+", " ", t).strip()
    return t in {"no response", "noresponse", "n/a", "na"}

def split_multiselect(text: str) -> list[str]:
    """
    복수선택 응답 파싱:
    구분자: ; , / | 줄바꿈
    """
    t = clean_text(text)
    if not t or is_no_response(t):
        return []
    t = t.replace("\n", ";").replace("•", ";")
    parts = re.split(r"[;,/|]+", t)
    return [p.strip() for p in parts if p.strip()]

def multiselect_summary_fixed(df: pd.DataFrame, col: str, option_order: list[str]) -> tuple[pd.DataFrame, int]:
    """
    ✅ 옵션 목록을 고정해 0도 포함하여 집계
    - 응답자수: 해당 옵션을 선택한 고유 응답자 수
    - 응답자비율: (응답자수 / (해당 문항에 1개 이상 응답한 사람 수)) * 100
    """
    base = df[[col]].copy()
    base["__rid__"] = base.index
    base["choices"] = base[col].apply(split_multiselect)

    total_respondents = (base["choices"].apply(len) > 0).sum()
    if total_respondents == 0:
        out0 = pd.DataFrame({"옵션": option_order, "응답자수": 0, "응답자비율(%)": 0.0})
        return out0, 0

    ex = base.explode("choices").dropna(subset=["choices"])
    ex["choices"] = ex["choices"].astype(str).str.strip()
    ex = ex[ex["choices"] != ""]

    # 설문 옵션과 완전 일치하는 것만 집계(옵션 외 문자열은 기타로 흡수할 수도 있음)
    allowed = set(option_order)
    ex.loc[~ex["choices"].isin(allowed), "choices"] = "기타"

    grp = ex.drop_duplicates(subset=["__rid__", "choices"]).groupby("choices")["__rid__"].nunique()

    # 0 포함 + 순서 고정
    out = pd.DataFrame({"옵션": option_order})
    out["응답자수"] = out["옵션"].map(grp).fillna(0).astype(int)
    out["응답자비율(%)"] = (out["응답자수"] / total_respondents * 100).round(2)
    return out, int(total_respondents)

def render_multi(col: str, other_col: str | None):
    st.markdown(f"#### {B_KEYWORDS.get(col, col)}")
    st.caption("복수선택 문항입니다. 그래프는 ‘응답자 기준 비율(%)’을 보여줍니다. (설문 옵션 목록 기준으로 0도 포함)")

    option_order = OPTIONS_MAP[col]
    summ, n_resp = multiselect_summary_fixed(fdf, col, option_order)

    st.metric("해당 문항 응답자 수 (N)", f"{n_resp:,}")

    # 그래프
    plot_df = summ.copy()
    fig = px.bar(
        plot_df,
        x="응답자비율(%)",
        y="옵션",
        orientation="h",
        text="응답자비율(%)",
        title="선택 비율 (응답자 기준 %)"
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis_title="응답자 비율(%)",
        yaxis_title=""
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("옵션별 빈도표")
    st.dataframe(summ, use_container_width=True, hide_index=True)

    # 기타 서술
    if other_col and other_col in fdf.columns:
        st.subheader("기타 (서술) 응답")
        other = fdf[other_col].astype(str).map(clean_text)
        other = other[(other != "") & (~other.map(is_no_response))]
        st.caption(f"기타 서술 응답 수 = {len(other):,}")
        if other.empty:
            st.info("기타 서술 응답이 없습니다.")
        else:
            st.dataframe(pd.DataFrame({"기타 응답": other}).head(200), use_container_width=True, hide_index=True)

# =========================
# UI
# =========================
st.markdown("### 🧩 공대-컴퓨터(ECS) 자유응답 문항 분석 (B1–B4)")
st.caption("B1–B3: 복수선택 빈도(응답자 기준 %) + 기타 서술, B4: 주관식 원문 + 키워드(참고용).")

df = load_data(CSV_URL_B)

# 필수 컬럼 체크
required = META_COLS + B_MULTI + [B_OPEN]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"CSV에 다음 컬럼이 없습니다: {missing}")
    st.stop()

# ECS만
ecs_df = df[df["Field_Group"].astype(str).str.strip().eq("ECS")].copy()
if ecs_df.empty:
    st.warning("ECS(Field_Group='ECS') 데이터가 없습니다. CSV를 확인해주세요.")
    st.stop()

with st.sidebar:
    st.header("필터 (Filters)")
    all_yl = sorted(ecs_df["Year_Level"].dropna().astype(str).unique().tolist())
    yl = st.multiselect("Year_Level", all_yl, default=all_yl)

    st.divider()
    show_raw = st.checkbox("원자료 일부 보기", value=False)

fdf = ecs_df[ecs_df["Year_Level"].isin(yl)].copy()

c1, c2 = st.columns(2)
c1.metric("표본 수 (ECS N)", f"{len(fdf):,}")
c2.metric("원 데이터 전체(ECS) N", f"{len(ecs_df):,}")

if show_raw:
    st.subheader("데이터 미리보기 (ECS)")
    candidate_cols = [COL_B1_OTHER, COL_B2_OTHER, COL_B3_OTHER, COL_B3_OTHER2]
    cols = META_COLS + B_MULTI + [c for c in candidate_cols if c in fdf.columns] + [B_OPEN]
    st.dataframe(fdf[cols].head(30), use_container_width=True)

# 탭
tab1, tab2, tab3, tab4 = st.tabs([
    "B1 [교육과정·정책 개선 요구]",
    "B2 [정서적 부담감의 원인]",
    "B3 [학습 내용 선호]",
    "B4 [대학에 요구사항]",
])

with tab1:
    render_multi(COL_B1, COL_B1_OTHER if COL_B1_OTHER in fdf.columns else None)

with tab2:
    render_multi(COL_B2, COL_B2_OTHER if COL_B2_OTHER in fdf.columns else None)

with tab3:
    # B3 기타 컬럼은 파일마다 다를 수 있어 우선순위로 선택
    b3_other = None
    if COL_B3_OTHER in fdf.columns:
        b3_other = COL_B3_OTHER
    elif COL_B3_OTHER2 in fdf.columns:
        b3_other = COL_B3_OTHER2
    render_multi(COL_B3, b3_other)

with tab4:
    st.markdown(f"#### {B_KEYWORDS.get(COL_B4, COL_B4)}")
    st.caption("주관식 문항입니다. 'No Response'는 제외됩니다.")

    open_s = fdf[COL_B4].astype(str).map(clean_text)
    open_s = open_s[(open_s != "") & (~open_s.map(is_no_response))]

    st.metric("주관식 응답 수 (N)", f"{len(open_s):,}")

    if open_s.empty:
        st.warning("주관식(B4) 응답이 없습니다.")
    else:
        st.subheader("주관식 원문 보기")
        q = st.text_input("원문 검색(포함 검색)", value="")
        view = open_s
        if q.strip():
            view = view[view.str.contains(q.strip(), na=False)]
        st.dataframe(pd.DataFrame({"B4 응답": view}).head(400), use_container_width=True, hide_index=True)
