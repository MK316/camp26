import streamlit as st

st.set_page_config(
    page_title="Survey Dashboard (Q01–Q12)",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("#### ❄️ Survey Dashboard (Q01–Q12): 참여 연구자 공유용")
st.write(
    """
이 대시보드는 1/20일에 마친 설문 데이터(Q01–Q12, 1–6 리커트)를 종합 요약하고,
학문영역(공대/인문/사범)/학년(졸업생)/세부전공에 따라 결과를 수치와 그래프로 확인할 수 있도록 구성되었습니다.

왼쪽 사이드바에서 메뉴를 선택하여 페이지를 이동하세요.  
각 페이지에 세부 탭(Tab)이 있고 다양한 정보를 보실 수 있습니다.
"""
)

st.markdown("""
    1. 🌀 개요 및 기술통계
    2. 🌀 항목별 결과
    3. 🌀 항목 상관분석
    4. 📮 Padlet link (현 프로젝트 파일공유 공간)
    """)


st.info("데이터는 GitHub CSV에서 불러옵니다. (캐시 적용)")

st.caption("5 PM, Jan. 20")
