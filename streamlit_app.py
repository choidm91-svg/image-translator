import streamlit as st
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
import pytesseract
from pytesseract import Output
import base64
import io
import re

st.set_page_config(
    page_title="AI 상세페이지 번역기",
    page_icon="🌐",
    layout="wide",
)

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("🌐 AI 상세페이지 번역기")
st.write("분할된 상세페이지 이미지를 여러 장 업로드하면 순서대로 번역합니다.")
st.caption("원본 이미지와 번역문은 별도 영역에 표시됩니다. 영문은 한국어 번역 대상으로 잡지 않도록 이중 OCR 필터를 적용합니다.")

language_map = {
    "러시아어": "Russian",
    "영어": "English",
    "일본어": "Japanese",
    "중국어": "Chinese",
    "베트남어": "Vietnamese",
}

selected_language = st.selectbox(
    "번역할 언어를 선택하세요",
    ["러시아어", "영어", "일본어", "중국어", "베트남어"],
)

uploaded_files = st.file_uploader(
    "분할된 JPG 또는 PNG 이미지를 여러 장 올려주세요",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

def image_to_base64(uploaded_file):
    uploaded_file.seek(0)
    image = Image.open(uploaded_file).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    uploaded_file.seek(0)
    return encoded, image

def pil_to_png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

def contains_korean(text):
    return bool(re.search(r"[가-힣]", text))

def count_hangul(text):
    return len(re.findall(r"[가-힣]", text))

def count_latin(text):
    return len(re.findall(r"[A-Za-z]", text))

def box_iou(a, b):
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = ax1 + a["w"], ay1 + a["h"]

    bx1, by1 = b["x"], b["y"]
    bx2, by2 = bx1 + b["w"], by1 + b["h"]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    inter = iw * ih
    area_a = max(1, a["w"] * a["h"])
    area_b = max(1, b["w"] * b["h"])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0

def get_confident_english_boxes(image):
    """
    같은 이미지를 영어 OCR로 한 번 더 읽습니다.
    영어로 자신 있게 읽히는 영역은 한국어 오인식 후보에서 제외합니다.
    """
    data = pytesseract.image_to_data(
        image,
        lang="eng",
        config="--oem 1 --psm 11",
        output_type=Output.DICT,
    )

    english_boxes = []

    for i, raw in enumerate(data["text"]):
        text = raw.strip()

        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1

        latin_count = count_latin(text)

        if conf < 55:
            continue

        if latin_count < 2:
            continue

        english_boxes.append(
            {
                "text": text,
                "x": int(data["left"][i]),
                "y": int(data["top"][i]),
                "w": int(data["width"][i]),
                "h": int(data["height"][i]),
                "confidence": conf,
            }
        )

    return english_boxes

def detect_korean_lines(image):
    """
    한국어 후보를 찾은 뒤,
    같은 위치가 영어 OCR에서 확실하게 영문으로 읽히면 제거합니다.

    또한 한 글자짜리 한글 후보와 영문 비율이 높은 줄을 제외해
    PACKAGE / Lifting / NEW / 17Hours 같은 영문이
    가짜 한글로 잡히는 것을 최대한 방지합니다.
    """

    english_boxes = get_confident_english_boxes(image)

    data = pytesseract.image_to_data(
        image,
        lang="kor+eng",
        config="--oem 1 --psm 11",
        output_type=Output.DICT,
    )

    groups = {}

    total = len(data["text"])

    for i in range(total):
        raw_text = data["text"][i].strip()

        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1

        if not raw_text or conf < 20:
            continue

        key = (
            data["block_num"][i],
            data["par_num"][i],
            data["line_num"][i],
        )

        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])

        if key not in groups:
            groups[key] = {
                "words": [],
                "x1": x,
                "y1": y,
                "x2": x + w,
                "y2": y + h,
                "conf_values": [],
            }

        groups[key]["words"].append(raw_text)
        groups[key]["x1"] = min(groups[key]["x1"], x)
        groups[key]["y1"] = min(groups[key]["y1"], y)
        groups[key]["x2"] = max(groups[key]["x2"], x + w)
        groups[key]["y2"] = max(groups[key]["y2"], y + h)
        groups[key]["conf_values"].append(conf)

    lines = []

    for group in groups.values():
        text = " ".join(group["words"]).strip()

        hangul_count = count_hangul(text)
        latin_count = count_latin(text)
        total_letters = hangul_count + latin_count

        if hangul_count == 0:
            continue

        # 영문을 '후', '비' 같은 한 글자 한글로 잘못 읽는 오탐 차단
        if hangul_count < 2:
            continue

        # 실제 한글보다 영문 비율이 더 높은 줄은 한국어 카피로 보지 않음
        hangul_ratio = hangul_count / max(1, total_letters)
        if hangul_ratio < 0.45:
            continue

        candidate = {
            "text": text,
            "x": group["x1"],
            "y": group["y1"],
            "w": group["x2"] - group["x1"],
            "h": group["y2"] - group["y1"],
        }

        # 영어 OCR이 같은 위치를 영문으로 확실하게 읽으면 한국어 후보 제거
        english_overlap = False

        for eng in english_boxes:
            if box_iou(candidate, eng) >= 0.30:
                english_overlap = True
                break

        if english_overlap:
            continue

        conf_values = group["conf_values"]
        avg_conf = sum(conf_values) / len(conf_values) if conf_values else 0

        candidate["confidence"] = round(avg_conf, 1)
        candidate["hangul_ratio"] = round(hangul_ratio, 2)

        lines.append(candidate)

    lines.sort(key=lambda item: (item["y"], item["x"]))
    return lines

def draw_detection_preview(image, lines):
    """
    원본 이미지를 복사해 한국어 OCR 박스만 표시합니다.
    원본 파일 자체는 수정하지 않습니다.
    """
    preview = image.copy()
    draw = ImageDraw.Draw(preview)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            max(16, int(image.width * 0.022)),
        )
    except Exception:
        font = ImageFont.load_default()

    line_width = max(2, int(image.width * 0.004))

    for index, line in enumerate(lines, start=1):
        x1 = line["x"]
        y1 = line["y"]
        x2 = x1 + line["w"]
        y2 = y1 + line["h"]

        pad = max(3, int(image.width * 0.004))

        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(image.width, x2 + pad)
        y2 = min(image.height, y2 + pad)

        draw.rectangle(
            [x1, y1, x2, y2],
            outline=(255, 0, 0),
            width=line_width,
        )

        label = str(index)
        label_x = x1
        label_y = max(0, y1 - max(22, int(image.width * 0.03)))

        bbox = draw.textbbox((label_x, label_y), label, font=font)
        bg_pad = 3

        draw.rectangle(
            [
                bbox[0] - bg_pad,
                bbox[1] - bg_pad,
                bbox[2] + bg_pad,
                bbox[3] + bg_pad,
            ],
            fill=(255, 255, 255),
        )

        draw.text(
            (label_x, label_y),
            label,
            fill=(255, 0, 0),
            font=font,
        )

    return preview

if "all_results" not in st.session_state:
    st.session_state.all_results = []

if "ocr_results" not in st.session_state:
    st.session_state.ocr_results = {}

if uploaded_files:
    st.subheader("업로드된 이미지")
    st.write(f"총 {len(uploaded_files)}장의 이미지가 업로드되었습니다.")
    st.info("업로드한 순서대로 번역됩니다. (1번 → 2번 → 3번...)")

    for idx, file in enumerate(uploaded_files, start=1):
        st.markdown(f"**{idx}번 이미지: {file.name}**")

    if st.button("🚀 AI 번역 시작", type="primary"):
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
3. 브랜드명, 제품명, 영문 제품명은 함부로 번역하지 마세요.
4. 전성분은 번역하지 말고 아래처럼 표시하세요:
[전성분]
영문 INCI 유지
5. 화장품 광고 문구는 치료, 완치, 재생 같은 의료적 표현으로 과장하지 마세요.
6. 원문 의미를 최대한 유지하면서 자연스럽게 번역하세요.
7. 같은 문구가 반복되어 보이면 한 번만 정리하세요.
8. 불필요한 설명은 쓰지 말고 한국어 원문과 번역 결과만 정리하세요.
"""

                response = client.responses.create(
                    model="gpt-5",
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": prompt,
                                },
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high",
                                },
                            ],
                        }
                    ],
                )

                all_results.append(
                    {
                        "index": idx,
                        "file_name": uploaded_file.name,
                        "image": preview_image,
                        "translation": response.output_text,
                        "language": selected_language,
                    }
                )

                progress.progress(idx / len(uploaded_files))

        st.session_state.all_results = all_results
        st.session_state.ocr_results = {}
        st.success("✅ 모든 이미지 번역이 완료되었습니다.")

if st.session_state.all_results:
    st.markdown("---")
    st.header("📑 이미지별 번역 결과")

    combined_text = ""

    for item in st.session_state.all_results:
        st.markdown("---")
        st.subheader(f"{item['index']}번 이미지 · {item['file_name']}")

        top_col1, top_col2 = st.columns(2)

        with top_col1:
            st.download_button(
                label=f"📥 {item['index']}번 번역문 다운로드",
                data=item["translation"],
                file_name=f"{item['index']:02d}_{item['language']}_translation.txt",
                mime="text/plain",
                key=f"download_original_{item['index']}",
                use_container_width=True,
            )

        with top_col2:
            find_boxes = st.button(
                f"🔎 {item['index']}번 한국어 위치 찾기",
                key=f"detect_{item['index']}",
                use_container_width=True,
            )

        if find_boxes:
            with st.spinner(
                f"{item['index']}번 이미지에서 한국어 위치를 찾고 있습니다..."
            ):
                try:
                    lines = detect_korean_lines(item["image"])
                    preview = draw_detection_preview(item["image"], lines)

                    st.session_state.ocr_results[item["index"]] = {
                        "lines": lines,
                        "preview": preview,
                    }

                    if lines:
                        st.success(
                            f"한국어 텍스트 영역 {len(lines)}개를 찾았습니다."
                        )
                    else:
                        st.warning(
                            "한국어 영역을 찾지 못했습니다. 작은 글자나 장식 글자는 OCR에서 빠질 수 있습니다."
                        )

                except Exception as exc:
                    st.error(
                        "OCR 실행에 실패했습니다. packages.txt에 Tesseract 한국어 언어팩이 설치되어 있는지 확인해 주세요."
                    )
                    st.code(str(exc))

        col_image, col_text = st.columns([1, 1], gap="large")

        with col_image:
            st.markdown("### 🖼️ 원본 이미지")
            st.image(item["image"], use_container_width=True)

        with col_text:
            st.markdown(f"### 🌐 {item['language']} 번역")

            edited_translation = st.text_area(
                "번역문 검수/수정",
                value=item["translation"],
                height=600,
                key=f"translation_edit_{item['index']}",
            )

            st.download_button(
                label="📥 수정한 번역문 다운로드",
                data=edited_translation,
                file_name=f"{item['index']:02d}_{item['language']}_final.txt",
                mime="text/plain",
                key=f"download_edit_{item['index']}",
                use_container_width=True,
            )

        if item["index"] in st.session_state.ocr_results:
            ocr_data = st.session_state.ocr_results[item["index"]]
            lines = ocr_data["lines"]
            preview = ocr_data["preview"]

            st.markdown("### 🔎 한국어 위치 검사 결과")
            st.caption(
                "빨간 박스는 OCR이 찾은 한국어 영역입니다. 아직 번역문을 이미지에 넣지 않습니다."
            )

            preview_col, list_col = st.columns([1.2, 1], gap="large")

            with preview_col:
                st.image(
                    preview,
                    caption="한국어 위치 미리보기",
                    use_container_width=True,
                )

                st.download_button(
                    label="📥 위치 검사 이미지 PNG 다운로드",
                    data=pil_to_png_bytes(preview),
                    file_name=f"{item['index']:02d}_ocr_preview.png",
                    mime="image/png",
                    key=f"download_ocr_preview_{item['index']}",
                    use_container_width=True,
                )

            with list_col:
                if lines:
                    for line_index, line in enumerate(lines, start=1):
                        st.markdown(
                            f"**{line_index}. {line['text']}**  \n"
                            f"위치: x={line['x']}, y={line['y']}  \n"
                            f"크기: {line['w']}×{line['h']}px  \n"
                            f"OCR 신뢰도: {line['confidence']}"
                        )
                else:
                    st.info("표시할 한국어 OCR 결과가 없습니다.")

        combined_text += (
            f"\n\n========== {item['index']}번 이미지 ==========\n"
            f"파일명: {item['file_name']}\n\n"
            f"{edited_translation}\n"
        )

    st.markdown("---")
    st.header("📚 전체 번역 결과")

    st.text_area(
        "전체 이미지 번역 모음",
        value=combined_text,
        height=600,
        key="combined_translation",
    )

    st.download_button(
        label="📥 전체 번역 결과 다운로드",
        data=combined_text,
        file_name=f"ALL_{selected_language}_translation.txt",
        mime="text/plain",
        key="download_all",
    )
