import streamlit as st
from openai import OpenAI
from PIL import Image
import base64
import io

st.set_page_config(
    page_title="AI 상세페이지 번역기",
    page_icon="🌐",
    layout="wide"
)

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("🌐 AI 상세페이지 번역기")
st.write("분할된 상세페이지 이미지를 여러 장 업로드하면 순서대로 번역합니다.")
st.write("※ 번역문은 이미지 위에 쓰지 않고, 따로 표시하므로 이미지와 글자가 겹치지 않습니다.")

language_map = {
    "러시아어": "Russian",
    "영어": "English",
    "일본어": "Japanese",
    "중국어": "Chinese",
    "베트남어": "Vietnamese"
}

selected_language = st.selectbox(
    "번역할 언어를 선택하세요",
    ["러시아어", "영어", "일본어", "중국어", "베트남어"]
)

uploaded_files = st.file_uploader(
    "분할된 JPG 또는 PNG 이미지를 여러 장 올려주세요",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

def image_to_base64(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode("utf-8"), image

if uploaded_files:

    st.subheader("업로드된 이미지")
    st.write(f"총 {len(uploaded_files)}장의 이미지가 업로드되었습니다.")
    st.info("업로드한 순서대로 번역됩니다. (1번 → 2번 → 3번...)")

    for idx, file in enumerate(uploaded_files, start=1):
        st.markdown(f"**{idx}번 이미지: {file.name}**")

    if st.button("🚀 AI 번역 시작"):

        all_results = []
        progress = st.progress(0)

        for idx, uploaded_file in enumerate(uploaded_files, start=1):

            with st.spinner(f"{idx}/{len(uploaded_files)} 번역 중..."):

                base64_image, preview_image = image_to_base64(uploaded_file)

                target_language = language_map[selected_language]

                prompt = f"""
이 이미지는 한국 화장품 상세페이지의 분할 이미지입니다.

작업 목표:
1. 이미지 안의 한국어 문구를 위에서 아래 순서대로 읽으세요.
2. 각 문구를 {target_language}로 자연스럽게 번역하세요.
3. 결과는 반드시 아래 형식으로 정리하세요.

[한국어]
원문

[{selected_language}]
번역문

규칙:
1. 이미지에 실제로 보이는 문구만 적으세요.
2. 숫자, %, ppm, ml, g, 날짜, 시험 수치는 원문 그대로 유지하세요.
3. 브랜드명/제품명/영문 제품명은 함부로 번역하지 마세요.
4. 전성분은 번역하지 말고 아래처럼 표시하세요:
   [전성분]
   영문 INCI 유지
5. 화장품 광고 문구는 치료/완치/재생 같은 의료적 표현으로 과장하지 마세요.
6. 설명문은 짧고 자연스럽게 번역하세요.
7. 같은 문구가 반복되어 보이면 한 번만 정리하세요.
8. 불필요한 설명은 쓰지 말고, 번역 결과만 정리하세요.
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
                                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            ]
                        }
                    ]
                )

                result_text = response.output_text
                all_results.append(
                    {
                        "index": idx,
                        "file_name": uploaded_file.name,
                        "image": preview_image,
                        "translation": result_text
                    }
                )

                progress.progress(idx / len(uploaded_files))

        st.success("모든 이미지 번역이 완료되었습니다.")

        st.subheader("번역 결과")

        combined_text = ""

        for item in all_results:
            st.markdown("---")
            st.markdown(f"## {item['index']}번 이미지 - {item['file_name']}")

            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(item["image"], caption=f"{item['index']}번 원본 이미지", use_container_width=True)

            with col2:
                st.text_area(
                    f"{item['index']}번 번역 결과",
                    item["translation"],
                    height=500,
                    key=f"result_{item['index']}"
                )

            combined_text += f"\n\n========== {item['index']}번 이미지 : {item['file_name']} ==========\n\n"
            combined_text += item["translation"]

        st.markdown("---")
        st.subheader("전체 번역 결과 모음")

        st.text_area(
            "전체 번역 결과",
            combined_text,
            height=500
        )

        st.download_button(
            "📥 전체 번역 결과 TXT 다운로드",
            data=combined_text,
            file_name="translated_result.txt",
            mime="text/plain"
        )
