import streamlit as st

st.set_page_config(
    page_title="Survey Dashboard (Q01–Q12)",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("#### ❄️ Survey Dashboard (Q01–Q12): 참여 연구자 공유용")
st.write(
    """
이 대시보드는 1/20일에 마친 설문 데이터(Q01–Q12, 1–6 리커트)를 요약하고,
필드/학년/전공 분야에 따라 결과를 수치와 그래프로 확인할 수 있도록 구성되었습니다.

왼쪽 사이드바에서 메뉴를 선택하여 페이지를 이동하세요.
"""
)

st.info("데이터는 GitHub CSV에서 불러옵니다. (캐시 적용)")
