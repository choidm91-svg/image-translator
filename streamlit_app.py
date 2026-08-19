import streamlit as st
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont, ImageOps
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
st.caption("한국어 위치 검출 시 영문 오인식 제거, 같은 줄 문장 병합, 흰색 한글 보완 검사를 함께 적용합니다.")

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

def intersection_area(a, b):
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

    return iw * ih

def box_iou(a, b):
    inter = intersection_area(a, b)

    area_a = max(1, a["w"] * a["h"])
    area_b = max(1, b["w"] * b["h"])

    union = area_a + area_b - inter
    return inter / union if union > 0 else 0

def overlap_ratio_of_candidate(candidate, other):
    """
    candidate 영역 중 other 박스가 얼마나 덮는지 계산합니다.
    PACKAGE RENEWAL처럼 영어 단어가 여러 조각으로 나뉘어도
    후보 영역을 많이 덮으면 영문으로 판단할 수 있게 합니다.
    """
    inter = intersection_area(candidate, other)
    area_candidate = max(1, candidate["w"] * candidate["h"])
    return inter / area_candidate

def vertical_overlap_ratio(a, b):
    ay1, ay2 = a["y"], a["y"] + a["h"]
    by1, by2 = b["y"], b["y"] + b["h"]

    overlap = max(0, min(ay2, by2) - max(ay1, by1))
    return overlap / max(1, min(a["h"], b["h"]))

def horizontal_gap(a, b):
    if a["x"] <= b["x"]:
        return b["x"] - (a["x"] + a["w"])
    return a["x"] - (b["x"] + b["w"])

def merge_two_boxes(a, b):
    x1 = min(a["x"], b["x"])
    y1 = min(a["y"], b["y"])
    x2 = max(a["x"] + a["w"], b["x"] + b["w"])
    y2 = max(a["y"] + a["h"], b["y"] + b["h"])

    if a["x"] <= b["x"]:
        merged_text = (a["text"].rstrip() + " " + b["text"].lstrip()).strip()
    else:
        merged_text = (b["text"].rstrip() + " " + a["text"].lstrip()).strip()

    return {
        "text": re.sub(r"\s+", " ", merged_text),
        "x": x1,
        "y": y1,
        "w": x2 - x1,
        "h": y2 - y1,
        "confidence": round(
            (float(a.get("confidence", 0)) + float(b.get("confidence", 0))) / 2,
            1,
        ),
    }

def merge_same_line_boxes(lines, image_width):
    """
    같은 줄인데 OCR이 '습니다', '니다', '제'처럼 여러 박스로
    쪼갠 결과를 하나의 문장 박스로 합칩니다.
    """
    if not lines:
        return []

    lines = sorted(lines, key=lambda item: (item["y"], item["x"]))
    merged = []

    for current in lines:
        if not merged:
            merged.append(current.copy())
            continue

        previous = merged[-1]

        v_overlap = vertical_overlap_ratio(previous, current)
        gap = horizontal_gap(previous, current)

        avg_h = (previous["h"] + current["h"]) / 2
        max_gap = max(
            35,
            int(avg_h * 3.2),
            int(image_width * 0.045),
        )

        # 같은 줄 + 간격이 가까우면 하나의 문장으로 병합
        if v_overlap >= 0.55 and -20 <= gap <= max_gap:
            merged[-1] = merge_two_boxes(previous, current)
        else:
            merged.append(current.copy())

    return merged

def deduplicate_boxes(lines):
    """
    원본/반전 OCR을 동시에 돌렸을 때 같은 문구가 중복 검출되는 것을 제거합니다.
    """
    result = []

    for line in sorted(lines, key=lambda item: (item["y"], item["x"])):
        duplicate_index = None

        for idx, existing in enumerate(result):
            if box_iou(line, existing) >= 0.55:
                duplicate_index = idx
                break

        if duplicate_index is None:
            result.append(line)
        else:
            existing = result[duplicate_index]

            # 더 긴 한국어 문장을 우선 사용
            if count_hangul(line["text"]) > count_hangul(existing["text"]):
                result[duplicate_index] = line
            elif line.get("confidence", 0) > existing.get("confidence", 0):
                result[duplicate_index] = line

    return result

def get_confident_english_boxes(image):
    """
    영어 전용 OCR 결과를 수집합니다.
    실제 영문을 한국어로 오인한 후보를 제거하는 데 사용합니다.
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

        if conf < 45:
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

def english_coverage(candidate, english_boxes):
    """
    후보 박스 내부를 영어 OCR 박스들이 얼마나 덮는지 계산합니다.
    영어 단어가 2~3개로 분리돼도 합산하여 판정합니다.
    """
    candidate_area = max(1, candidate["w"] * candidate["h"])
    covered = 0

    for eng in english_boxes:
        covered += intersection_area(candidate, eng)

    # 겹치는 영어 박스끼리 중복 계산될 수 있으므로 1.0으로 제한
    return min(1.0, covered / candidate_area)

def run_korean_candidate_ocr(image):
    """
    한 번의 OCR 패스에서 한국어 후보 라인을 추출합니다.
    """
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

        if not raw_text or conf < 18:
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
        text = re.sub(r"\s+", " ", " ".join(group["words"]).strip())

        hangul_count = count_hangul(text)
        latin_count = count_latin(text)

        if hangul_count == 0:
            continue

        total_letters = hangul_count + latin_count
        hangul_ratio = hangul_count / max(1, total_letters)

        # 영어를 '후', '비' 등으로 잘못 읽은 짧은 오탐 제거
        if hangul_count == 1 and len(text.replace(" ", "")) > 1:
            continue

        # 영문이 대부분인 줄은 한국어 본문이 아님
        if hangul_ratio < 0.40:
            continue

        conf_values = group["conf_values"]
        avg_conf = sum(conf_values) / len(conf_values) if conf_values else 0

        lines.append(
            {
                "text": text,
                "x": group["x1"],
                "y": group["y1"],
                "w": group["x2"] - group["x1"],
                "h": group["y2"] - group["y1"],
                "confidence": round(avg_conf, 1),
            }
        )

    return lines

def detect_korean_lines(image):
    """
    개선된 한국어 위치 검출

    1. 원본 이미지 OCR
    2. 명암 반전 이미지 OCR
       -> '기존', '리뉴얼'처럼 색 박스 안 흰색 글자 검출 보완
    3. 영어 전용 OCR과 비교
       -> PACKAGE RENEWAL / Lifting / NEW / 17Hours 오탐 제거
    4. 같은 줄에서 쪼개진 문장을 하나의 박스로 병합
    """

    english_boxes = get_confident_english_boxes(image)

    # 첫 번째 패스: 원본
    candidates = run_korean_candidate_ocr(image)

    # 두 번째 패스: 대비 강화 + 반전
    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray)
    inverted = ImageOps.invert(gray).convert("RGB")

    inverted_candidates = run_korean_candidate_ocr(inverted)

    candidates.extend(inverted_candidates)
    candidates = deduplicate_boxes(candidates)

    filtered = []

    for candidate in candidates:
        hangul_count = count_hangul(candidate["text"])
        latin_count = count_latin(candidate["text"])
        hangul_ratio = hangul_count / max(1, hangul_count + latin_count)

        coverage = english_coverage(candidate, english_boxes)

        # PACKAGE RENEWAL처럼 영어 OCR이 후보 대부분을 덮으면 제거
        if coverage >= 0.38 and hangul_ratio < 0.80:
            continue

        # 영어 전용 OCR과 강하게 겹치고 한글이 매우 짧으면 제거
        strongly_english = False

        for eng in english_boxes:
            if overlap_ratio_of_candidate(candidate, eng) >= 0.45:
                if hangul_count <= 3:
                    strongly_english = True
                    break

        if strongly_english:
            continue

        filtered.append(candidate)

    # 먼저 같은 줄 분절을 합친 뒤 정렬
    filtered = merge_same_line_boxes(filtered, image.width)
    filtered = deduplicate_boxes(filtered)
    filtered.sort(key=lambda item: (item["y"], item["x"]))

    return filtered

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
