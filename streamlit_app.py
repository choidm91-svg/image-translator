import base64
import io
import json
import re

import streamlit as st
from openai import OpenAI
from PIL import Image

st.set_page_config(page_title="AI 상세페이지 번역기 v11", page_icon="🌐", layout="wide")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

LANGUAGE_MAP = {
    "러시아어": "Russian",
    "영어": "English",
    "일본어": "Japanese",
    "중국어": "Chinese",
    "베트남어": "Vietnamese",
    "프랑스어": "French",
    "스페인어": "Spanish",
}

st.title("🌐 AI 상세페이지 번역기 v11")
st.caption("실사용 우선: OCR 좌표 검출 없이 AI가 상세페이지 전체를 직접 읽고 한국어 원문 + 번역문을 정리합니다.")

selected_language = st.selectbox("번역할 언어", list(LANGUAGE_MAP.keys()), index=0)
safety_mode = st.checkbox("화장품 광고 표현을 보수적으로 번역", value=True)

uploaded_files = st.file_uploader(
    "상세페이지 JPG / PNG 이미지를 여러 장 업로드하세요",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)


def image_to_data_url(uploaded_file):
    uploaded_file.seek(0)
    image = Image.open(uploaded_file).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    uploaded_file.seek(0)
    return f"data:image/jpeg;base64,{encoded}", image


def clean_json_text(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\\s*", "", text)
        text = re.sub(r"\\s*```$", "", text)
    return text.strip()


def parse_json_response(text):
    cleaned = clean_json_text(text)
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    obj = re.search(r"\\{.*\\}", cleaned, re.S)
    if obj:
        try:
            return json.loads(obj.group(0))
        except Exception:
            return None
    return None


def translate_detail_image(image_url, target_language, safety_mode):
    safety_instruction = (
        "치료, 완치, 재생, 의약품 효능처럼 보일 수 있는 표현은 원문 의미를 유지하는 범위에서만 보수적으로 번역하세요."
        if safety_mode
        else
        "원문 의미를 최대한 그대로 유지하세요."
    )

    prompt = f"""
이 이미지는 한국 화장품 상세페이지입니다.

이미지 전체를 위에서 아래, 왼쪽에서 오른쪽 순서로 읽고
실제로 보이는 한국어만 추출한 뒤 {target_language}로 번역하세요.

반드시 JSON만 반환하세요.

형식:
{{
  "segments": [
    {{
      "order": 1,
      "korean": "한국어 원문",
      "translation": "{target_language} 번역",
      "type": "headline | body | label | footnote | test_value | ingredients | other"
    }}
  ]
}}

규칙:
1. 실제로 보이는 한국어만 적으세요.
2. 작은 글자, 각주, 사용 전/사용 후, 표 안의 한국어, 테스트 설명도 확인하세요.
3. 영어만 있는 브랜드명/제품명/패키지 영문은 새로 번역하지 마세요.
4. 숫자, %, ppm, ml, g, 날짜, 시험 수치, 별표(*)는 그대로 유지하세요.
5. 실제 전체 전성분 목록은 번역하지 마세요. 전체 전성분이 보이면 korean="전성분", translation="영문 INCI 유지", type="ingredients" 로 한 번만 적으세요.
6. 제품 패키지에 인쇄된 글자나 사진 속 문서의 작은 인쇄물은 상세페이지 카피가 아니라면 제외하세요.
7. 같은 문구를 중복해서 적지 마세요.
8. 여러 줄로 나뉜 하나의 문장은 합쳐서 적으세요.
9. 원문을 과도하게 의역하지 마세요.
10. {safety_instruction}
11. 설명/해설 없이 JSON만 반환하세요.
"""

    response = client.responses.create(
        model="gpt-5",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_url, "detail": "high"},
                ],
            }
        ],
    )

    parsed = parse_json_response(response.output_text)
    segments = parsed.get("segments", []) if isinstance(parsed, dict) else []

    cleaned = []
    for idx, seg in enumerate(segments, start=1):
        if not isinstance(seg, dict):
            continue
        korean = str(seg.get("korean", "")).strip()
        translation = str(seg.get("translation", "")).strip()
        seg_type = str(seg.get("type", "other")).strip() or "other"
        if not korean and not translation:
            continue
        cleaned.append({"order": idx, "korean": korean, "translation": translation, "type": seg_type})

    return cleaned, response.output_text


def segments_to_text(file_name, language, segments):
    parts = [f"===== {file_name} =====", f"번역 언어: {language}", ""]
    for seg in segments:
        parts.extend([
            f"[{seg['order']}]",
            "[한국어]",
            seg["korean"],
            "",
            f"[{language}]",
            seg["translation"],
            "",
        ])
    return "\\n".join(parts)


if "v11_results" not in st.session_state:
    st.session_state.v11_results = []

if uploaded_files:
    st.info("이 버전은 위치 박스 검수 없이 번역 자체를 빠르게 끝내는 실사용 우선 모드입니다.")

    if st.button("🚀 전체 이미지 번역 시작", type="primary"):
        results = []
        progress = st.progress(0)
        target_language = LANGUAGE_MAP[selected_language]

        for idx, uploaded_file in enumerate(uploaded_files, start=1):
            with st.spinner(f"{idx}/{len(uploaded_files)} · {uploaded_file.name} 번역 중..."):
                image_url, image = image_to_data_url(uploaded_file)
                try:
                    segments, raw_response = translate_detail_image(image_url, target_language, safety_mode)
                    results.append({
                        "index": idx,
                        "file_name": uploaded_file.name,
                        "image": image,
                        "segments": segments,
                        "raw_response": raw_response,
                    })
                except Exception as exc:
                    results.append({
                        "index": idx,
                        "file_name": uploaded_file.name,
                        "image": image,
                        "segments": [],
                        "raw_response": "",
                        "error": str(exc),
                    })
            progress.progress(idx / len(uploaded_files))

        st.session_state.v11_results = results
        st.success("✅ 번역 완료")

if st.session_state.v11_results:
    st.markdown("---")
    st.header("📑 이미지별 번역 결과")
    all_text_parts = []

    for item in st.session_state.v11_results:
        st.markdown("---")
        st.subheader(f"{item['index']}번 · {item['file_name']}")
        col_image, col_result = st.columns([0.9, 1.2], gap="large")

        with col_image:
            st.image(item["image"], caption="원본 이미지", use_container_width=True)

        with col_result:
            if item.get("error"):
                st.error("번역 중 오류가 발생했습니다.")
                st.code(item["error"])
                continue

            segments = item["segments"]
            if not segments:
                st.warning("구조화된 번역 결과를 만들지 못했습니다. AI 원본 응답을 확인하세요.")
                st.text_area(
                    "AI 원본 응답",
                    value=item.get("raw_response", ""),
                    height=450,
                    key=f"raw_{item['index']}",
                )
                continue

            edited_segments = []
            for seg_idx, segment in enumerate(segments, start=1):
                st.markdown(f"**{seg_idx}. {segment['type']}**")
                korean_text = st.text_area(
                    "한국어 원문",
                    value=segment["korean"],
                    height=85,
                    key=f"ko_{item['index']}_{seg_idx}",
                )
                translated_text = st.text_area(
                    f"{selected_language} 번역",
                    value=segment["translation"],
                    height=100,
                    key=f"tr_{item['index']}_{seg_idx}",
                )
                edited_segments.append({
                    "order": seg_idx,
                    "korean": korean_text,
                    "translation": translated_text,
                    "type": segment["type"],
                })

            item_text = segments_to_text(item["file_name"], selected_language, edited_segments)
            st.download_button(
                label=f"📥 {item['index']}번 번역 TXT 다운로드",
                data=item_text,
                file_name=f"{item['index']:02d}_{selected_language}_translation.txt",
                mime="text/plain",
                key=f"download_{item['index']}",
                use_container_width=True,
            )
            all_text_parts.append(item_text)

    if all_text_parts:
        st.markdown("---")
        st.header("📚 전체 번역 결과")
        all_text = "\\n\\n".join(all_text_parts)
        st.text_area("전체 번역 모음", value=all_text, height=500, key="v11_all_text")
        st.download_button(
            label="📥 전체 번역 결과 TXT 다운로드",
            data=all_text,
            file_name=f"ALL_{selected_language}_translation.txt",
            mime="text/plain",
            use_container_width=True,
        )
