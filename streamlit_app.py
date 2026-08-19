import streamlit as st

st.set_page_config(
    page_title="AI 상세페이지 번역기",
    page_icon="🌐",
    layout="wide"
)

st.title("🌐 AI 상세페이지 번역기")

st.write(
    "화장품 상세페이지 이미지를 올리면 "
    "AI가 번역해주는 프로그램입니다."
)

uploaded_file = st.file_uploader(
    "JPG 또는 PNG 상세페이지를 올려주세요",
    type=["jpg", "jpeg", "png"]
)

language = st.selectbox(
    "번역할 언어",
    [
        "러시아어",
        "영어",
        "일본어",
        "중국어"
    ]
)

if uploaded_file is not None:

    st.success("이미지 업로드 완료!")

    st.image(
        uploaded_file,
        caption="원본 상세페이지"
    )

    if st.button("AI 번역 시작"):
        st.info(
            f"{language} 번역 기능을 다음 단계에서 연결합니다."
        )
