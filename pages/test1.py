import re
from collections import Counter

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# =========================================
# Education(사범) E-items (E1~E4)
# - E1~E3: 복수선택(옵션형)
# - E4: 주관식
# =========================================
st.set_page_config(page_title="Education E-items (E1–E4)", layout="wide")

CSV_URL_EDU = "https://raw.githubusercontent.com/MK316/camp26/refs/heads/main/data/Edu-essay123.csv"

# =========================
# Columns (actual CSV)
# =========================
META_COLS = ["Academic_Field", "Year_Level", "Year_Original"]

COL_E1 = "E1"
COL_E2 = "E2"
COL_E3 = "E3"
COL_E4 = "E4"

E_MULTI = [COL_E1, COL_E2, COL_E3]
E_OPEN = COL_E4

# =========================
# Display labels (shown on screen)
# =========================
DISPLAY_LABELS = {
    COL_E1: "E1. [교육과정·정책 개선 요구]",
    COL_E2: "E2. [정서적 부담감의 원인]",
    COL_E3: "E3. [학습 내용 선호]",
    COL_E4: "E4. [주관식: 자유롭게 기술]",
    "Academic_Field": "학문 분야 (Academic_Field)",
    "Year_Level": "학년 (Year_Level)",
    "Year_Original": "원 학년 표기 (Year_Original)",
}

# =========================================
# (선택) 옵션 목록 고정(원하면 리스트로 넣기)
# - None이면 데이터에서 자동 추출(권장: 우선 자동)
# =========================================
E1_OPTIONS = None
E2_OPTIONS = None
E3_OPTIONS = None

OPTIONS_MAP = {
    COL_E1: E1_OPTIONS,
    COL_E2: E2_OPTIONS,
    COL_E3: E3_OPTIONS,
}

# =========================================
# Helpers
# =========================================
@st.cache_data(show_spinner=False)
def load_data(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(url, encoding="cp949")

    # 문자열 공백 제거
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()

    # 필수 컬럼 체크 (E1~E4가 이미 존재해야 함)
    required = META_COLS + [COL_E1, COL_E2, COL_E3, COL_E4]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error("CSV에 다음 컬럼이 없습니다:\n\n- " + "\n- ".join(missing))
        st.stop()

    return df


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s).strip()


def is_no_response(text: str) -> bool:
    t = clean_text(str(text)).lower()
    if t in {"", "nan", "none"}:
        return True
    return t in {"no response", "noresponse", "n/a", "na"}


def split_multiselect(text: str) -> list[str]:
    """
    복수선택 응답 파싱:
    - 구분자: ; , / | 줄바꿈
    """
    t = clean_text(text)
    if not t or is_no_response(t):
        return []
    t = t.replace("\n", ";").replace("•", ";")
    parts = re.split(r"[;,/|]+", t)
    return [p.strip() for p in parts if p.strip()]


def multiselect_summary(df: pd.DataFrame, col: str, option_order: list[str] | None) -> tuple[pd.DataFrame, int]:
    base = df[[col]].copy()
    base["__rid__"] = base.index
    base["choices"] = base[col].apply(split_multiselect)

    n_resp = int((base["choices"].apply(len) > 0).sum())
    if n_resp == 0:
        out0 = pd.DataFrame({"옵션": (option_order or []), "응답자수": 0, "응답자비율(%)": 0.0})
        return out0, 0

    ex = base.explode("choices").dropna(subset=["choices"])
    ex["choices"] = ex["choices"].astype(str).str.strip()
    ex = ex[ex["choices"] != ""]

    # 옵션 목록 고정이 있으면, 옵션 외는 '기타'로 흡수
    if option_order:
        allowed = set(option_order)
        ex.loc[~ex["choices"].isin(allowed), "choices"] = "기타"
        if "기타" not in allowed:
            option_order = option_order + ["기타"]

    grp = ex.drop_duplicates(subset=["__rid__", "choices"]).groupby("choices")["__rid__"].nunique()

    # 자동(데이터 기반)일 때는 빈도 순으로 보기 좋게
    if option_order is None:
        options = grp.sort_values(ascending=False).index.tolist()
    else:
        options = option_order

    out = pd.DataFrame({"옵션": options})
    out["응답자수"] = out["옵션"].map(grp).fillna(0).astype(int)
    out["응답자비율(%)"] = (out["응답자수"] / n_resp * 100).round(2)

    # ✅ 많은 빈도 먼저
    out = out.sort_values(["응답자수", "옵션"], ascending=[False, True]).reset_index(drop=True)
    return out, n_resp


def tokenize_ko_basic(text: str, stop: set[str]) -> list[str]:
    t = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", str(text))
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return []
    toks = []
    for w in t.split(" "):
        if len(w) < 2:
            continue
        if w in stop:
            continue
        toks.append(w)
    return toks


# =========================================
# UI
# =========================================
st.markdown("### 🧩 사범(Education) 영역: E1–E4")
st.caption("E1–E3: 복수선택(응답자 기준 %), E4: 주관식(키워드/그룹 비교/공동출현 네트워크 + 워드클라우드(가능 시))")

df = load_data(CSV_URL_EDU)

with st.sidebar:
    st.header("필터 (Filters)")

    all_af = sorted(df["Academic_Field"].dropna().astype(str).unique().tolist())
    af = st.multiselect("학문 분야 (Academic_Field)", all_af, default=all_af)

    all_yl = sorted(df["Year_Level"].dropna().astype(str).unique().tolist())
    yl = st.multiselect("학년 (Year_Level)", all_yl, default=all_yl)

    all_yo = sorted(df["Year_Original"].dropna().astype(str).unique().tolist())
    yo = st.multiselect("원 학년 표기 (Year_Original)", all_yo, default=all_yo)

    st.divider()
    palette = st.selectbox(
        "색상 팔레트 (Bar 공통)",
        ["Plotly", "D3", "G10", "T10", "Alphabet", "Dark24", "Set2", "Pastel"],
        index=0,
    )
    show_raw = st.checkbox("원자료 일부 보기", value=False)

fdf = df[
    df["Academic_Field"].isin(af) &
    df["Year_Level"].isin(yl) &
    df["Year_Original"].isin(yo)
].copy()

c1, c2, c3 = st.columns(3)
c1.metric("표본 수 (현재 필터 N)", f"{len(fdf):,}")
c2.metric("선택 Academic_Field 수", f"{len(af):,}")
c3.metric("선택 Year_Level 수", f"{len(yl):,}")

if show_raw:
    st.subheader("데이터 미리보기")
    show_cols = META_COLS + [COL_E1, COL_E2, COL_E3, COL_E4]
    st.dataframe(fdf[show_cols].rename(columns=DISPLAY_LABELS).head(30), use_container_width=True)

tab1, tab2, tab3, tab4 = st.tabs([
    DISPLAY_LABELS[COL_E1],
    DISPLAY_LABELS[COL_E2],
    DISPLAY_LABELS[COL_E3],
    DISPLAY_LABELS[COL_E4],
])

# 팔레트 -> 색상 시퀀스
color_seq = getattr(px.colors.qualitative, palette, px.colors.qualitative.Plotly)


def render_multi(col: str, option_order: list[str] | None):
    st.markdown(f"#### {DISPLAY_LABELS.get(col, col)}")
    st.caption("복수선택 문항입니다. 그래프는 ‘응답자 기준 비율(%)’을 보여줍니다.")

    summ, n_resp = multiselect_summary(fdf, col, option_order)
    st.metric("해당 문항 응답자 수 (N)", f"{n_resp:,}")

    if n_resp == 0:
        st.info("현재 필터 조건에서 유효 응답이 없습니다.")
        return

    # 항목별 색(팔레트 적용)
    opts = summ["옵션"].tolist()
    cmap = {opt: color_seq[i % len(color_seq)] for i, opt in enumerate(opts)}

    # 가로막대: 위쪽에 큰 값
    plot_df = summ.sort_values("응답자비율(%)", ascending=True)

    fig = px.bar(
        plot_df,
        x="응답자비율(%)",
        y="옵션",
        orientation="h",
        text="응답자비율(%)",
        color="옵션",
        color_discrete_map=cmap,
        title="선택 비율 (응답자 기준 %)"
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
    fig.update_layout(
        height=560,
        showlegend=False,
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis_title="응답자 비율(%)",
        yaxis_title=""
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("옵션별 빈도표")
    st.dataframe(summ, use_container_width=True, hide_index=True)


with tab1:
    render_multi(COL_E1, OPTIONS_MAP[COL_E1])

with tab2:
    render_multi(COL_E2, OPTIONS_MAP[COL_E2])

with tab3:
    render_multi(COL_E3, OPTIONS_MAP[COL_E3])

with tab4:
    st.markdown(f"#### {DISPLAY_LABELS.get(COL_E4, COL_E4)}")
    st.caption("주관식 문항입니다. 'No Response'는 제외됩니다. 아래는 키워드 빈도, 그룹별 비교, 공동출현 네트워크, 워드클라우드를 포함합니다.")

    open_s = fdf[COL_E4].astype(str).map(clean_text)
    open_s = open_s[(open_s != "") & (~open_s.map(is_no_response))]

    st.metric("주관식 응답 수 (N)", f"{len(open_s):,}")
    if open_s.empty:
        st.warning("현재 필터 조건에서 E4 주관식 응답이 없습니다.")
        st.stop()

    STOP = {
        "그리고","하지만","또한","그래서","때문","정도","같아요","합니다","했다","하는","에서","으로","에게",
        "것","수","등","좀","더","제","저","우리","너무","정말","있다","없다","이다","되다","있는",
        "합니다","됩니다","하는데","하면","해서","하여","대한","관련","필요","중요","우선","제공"
    }

    doc_tokens = [tokenize_ko_basic(x, STOP) for x in open_s.tolist()]
    all_tokens = [t for toks in doc_tokens for t in toks]

    # (A) 키워드 빈도
    st.subheader("🔎 전체 상위 키워드")
    top_n = st.slider("Top 키워드 개수", 10, 120, 40, 5, key="edu_e4_topn")

    if not all_tokens:
        st.info("키워드를 추출할 텍스트가 충분하지 않습니다.")
    else:
        freq = Counter(all_tokens)
        freq_df = pd.DataFrame(freq.most_common(top_n), columns=["keyword", "count"])
        fig_kw = px.bar(
            freq_df.sort_values("count", ascending=True),
            x="count", y="keyword",
            orientation="h",
            title=f"전체 상위 {top_n}개 키워드"
        )
        fig_kw.update_layout(height=680, margin=dict(l=20, r=20, t=60, b=20),
                             xaxis_title="빈도", yaxis_title="키워드")
        st.plotly_chart(fig_kw, use_container_width=True)
        st.dataframe(freq_df, use_container_width=True, hide_index=True)

    # (B) 그룹별 키워드 비교
    st.subheader("👥 그룹별 키워드 비교")
    group_col = st.selectbox("그룹 기준 선택", ["Academic_Field", "Year_Level", "Year_Original"], index=0, key="edu_e4_groupcol")
    min_n = st.slider("그룹 최소 응답 수", 1, 30, 5, key="edu_e4_min_group_n")

    tmp_df = fdf.copy()
    tmp_df["__open__"] = tmp_df[COL_E4].astype(str).map(clean_text)
    tmp_df = tmp_df[(tmp_df["__open__"] != "") & (~tmp_df["__open__"].map(is_no_response))]

    grp_counts = tmp_df.groupby(group_col)["__open__"].count().reset_index(name="N")
    valid_groups = grp_counts[grp_counts["N"] >= min_n][group_col].astype(str).tolist()

    if not valid_groups:
        st.info("현재 조건에서 최소 응답 수 기준을 만족하는 그룹이 없습니다. (최소 응답 수를 낮춰보세요.)")
    else:
        show_groups = st.multiselect(
            "표시할 그룹 선택",
            valid_groups,
            default=valid_groups[: min(6, len(valid_groups))],
            key="edu_e4_groups_pick"
        )
        per_top = st.slider("그룹별 Top 키워드 수", 5, 40, 12, 1, key="edu_e4_per_top")

        rows = []
        for gname in show_groups:
            sub_text = tmp_df[tmp_df[group_col].astype(str) == str(gname)]["__open__"].tolist()
            toks = [t for text in sub_text for t in tokenize_ko_basic(text, STOP)]
            if not toks:
                continue
            for kw, ct in Counter(toks).most_common(per_top):
                rows.append({"Group": str(gname), "keyword": kw, "count": ct})

        if not rows:
            st.info("선택한 그룹에서 추출된 키워드가 없습니다.")
        else:
            gkw = pd.DataFrame(rows)
            fig_gkw = px.bar(
                gkw,
                x="count",
                y="keyword",
                color="Group",
                orientation="h",
                title=f"{group_col}별 상위 키워드 비교 (Top {per_top})"
            )
            fig_gkw.update_layout(height=740, margin=dict(l=20, r=20, t=60, b=20),
                                  xaxis_title="빈도", yaxis_title="키워드")
            st.plotly_chart(fig_gkw, use_container_width=True)
            st.dataframe(gkw.sort_values(["Group", "count"], ascending=[True, False]),
                         use_container_width=True, hide_index=True)

    # (C) 공동출현 네트워크
    st.subheader("🕸️ 키워드 공동출현 네트워크")
    st.caption("한 응답 안에서 함께 등장한 키워드 쌍을 연결합니다. (상위 키워드 중심)")

    net_top = st.slider("네트워크에 포함할 상위 키워드 수", 10, 150, 50, 5, key="edu_e4_net_top")
    min_edge = st.slider("엣지 최소 공동출현 횟수", 1, 20, 2, 1, key="edu_e4_net_min_edge")

    if not all_tokens:
        st.info("네트워크를 만들 토큰이 없습니다.")
    else:
        import numpy as np

        top_vocab = [k for k, _ in Counter(all_tokens).most_common(net_top)]
        vocab_set = set(top_vocab)

        pair_counter = Counter()
        for toks in doc_tokens:
            uniq = [t for t in set(toks) if t in vocab_set]
            uniq.sort()
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    pair_counter[(uniq[i], uniq[j])] += 1

        edges = [(a, b, w) for (a, b), w in pair_counter.items() if w >= min_edge]

        if not edges:
            st.info("현재 설정(min_edge 등)에서 네트워크 엣지가 없습니다. 엣지 최소 공동출현 횟수를 낮춰보세요.")
        else:
            node_w = {k: Counter(all_tokens)[k] for k in top_vocab}

            n = len(top_vocab)
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            pos = {top_vocab[i]: (np.cos(angles[i]), np.sin(angles[i])) for i in range(n)}

            edge_x, edge_y = [], []
            for a, b, w in edges:
                x0, y0 = pos[a]
                x1, y1 = pos[b]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                mode="lines",
                hoverinfo="none",
                line=dict(width=1),
                name="co-occurrence"
            )

            node_x = [pos[k][0] for k in top_vocab]
            node_y = [pos[k][1] for k in top_vocab]
            node_size = [max(10, min(38, node_w[k])) for k in top_vocab]
            node_text = [f"{k} (freq={node_w[k]})" for k in top_vocab]

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode="markers+text",
                text=top_vocab,
                textposition="top center",
                hovertext=node_text,
                hoverinfo="text",
                marker=dict(size=node_size),
                name="keywords"
            )

            fig_net = go.Figure(data=[edge_trace, node_trace])
            fig_net.update_layout(
                title="키워드 공동출현 네트워크 (원형 배치)",
                showlegend=False,
                height=780,
                margin=dict(l=10, r=10, t=60, b=10),
                xaxis=dict(visible=False),
                yaxis=dict(visible=False)
            )
            st.plotly_chart(fig_net, use_container_width=True)

            edge_df = pd.DataFrame(edges, columns=["keyword_a", "keyword_b", "cooccur"])
            edge_df = edge_df.sort_values("cooccur", ascending=False).head(200)
            st.subheader("공동출현 상위 엣지(Top 200)")
            st.dataframe(edge_df, use_container_width=True, hide_index=True)

    # (D) 워드클라우드 (가능한 경우)
    st.subheader("☁️ 워드클라우드 (가능한 경우)")
    st.caption("서버에 wordcloud 패키지가 없으면 자동으로 건너뜁니다. 한글 폰트가 필요합니다.")

    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt

        font_path = "assets/NanumGothic-Regular.ttf"

        if not all_tokens:
            st.info("워드클라우드를 만들 토큰이 없습니다.")
        else:
            freq_dict = dict(Counter(all_tokens))
            wc = WordCloud(
                font_path=font_path,
                width=1400,
                height=650,
                background_color="white",
                prefer_horizontal=0.9
            ).generate_from_frequencies(freq_dict)

            fig, ax = plt.subplots(figsize=(14, 6.5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig, clear_figure=True)

    except ModuleNotFoundError:
        st.info("wordcloud 패키지가 없어 워드클라우드를 표시할 수 없습니다. requirements.txt에 wordcloud를 추가하면 됩니다.")
    except FileNotFoundError:
        st.error("폰트 파일을 찾을 수 없습니다. assets/NanumGothic-Regular.ttf 경로를 확인하세요.")
    except Exception as e:
        st.warning(f"워드클라우드 생성 중 오류: {e}")
