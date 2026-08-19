import streamlit as st
from openai import OpenAI
import base64

st.set_page_config(
    page_title="AI 상세페이지 번역기",
    page_icon="🌐",
    layout="wide"
)

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("🌐 AI 상세페이지 번역기")
st.write("화장품 상세페이지를 올리면 한국어를 읽고 번역합니다.")

uploaded_file = st.file_uploader(
    "JPG 또는 PNG 상세페이지를 올려주세요",
    type=["jpg", "jpeg", "png"]
)

language = st.selectbox(
    "번역할 언어",
    ["러시아어", "영어", "일본어", "중국어"]
)

if uploaded_file is not None:

    st.image(uploaded_file, caption="원본 상세페이지")

    if st.button("AI 번역 시작"):

        with st.spinner("이미지의 글자를 읽고 번역하고 있습니다..."):

            image_bytes = uploaded_file.getvalue()
            base64_image = base64.b64encode(image_bytes).decode("utf-8")

            mime_type = uploaded_file.type

            prompt = f"""
이 이미지는 한국 화장품 상세페이지입니다.

이미지 안의 한국어 문구를 위에서 아래 순서대로 정확하게 읽으세요.
그 다음 각 문장을 {language}로 자연스럽게 번역하세요.

반드시 다음 형식으로 출력하세요.

[한국어]
원문

[{language}]
번역문

규칙:
1. 한국어 원문을 빠뜨리지 마세요.
2. 숫자, %, ppm, 시험 수치, 날짜는 원문 그대로 유지하세요.
3. 브랜드명과 제품명은 함부로 번역하지 마세요.
4. 화장품 광고 문구는 치료 효과처럼 과장해서 번역하지 마세요.
5. 전성분이 나오면 임의 번역하지 말고 별도로 표시하세요.
6. 이미지 위쪽부터 아래쪽 순서대로 정리하세요.
"""

            response = client.responses.create(
                model="gpt-5",
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": prompt
                            },
                            {
                                "type": "input_image",
                                "image_url": f"data:{mime_type};base64,{base64_image}",
                                "detail": "high"
                            }
                        ]
                    }
                ]
            )

            st.subheader("번역 결과")
            st.write(response.output_text)
