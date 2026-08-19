import streamlit as st
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont, ImageOps
import pytesseract
from pytesseract import Output
import base64
import io
import re
import json

st.set_page_config(
    page_title="AI 상세페이지 번역기",
    page_icon="🌐",
    layout="wide",
)

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("🌐 AI 상세페이지 번역기")
st.write("분할된 상세페이지 이미지를 여러 장 업로드하면 순서대로 번역합니다.")
st.caption(
    "OCR v7: 2배 확대 OCR + 좌우 컬럼 병합 방지 + AI 최종 분류 "
    "(상세페이지 카피 / 전성분 / 삽입 이미지 / 영문 / 노이즈)"
)

language_map = {
    "러시아어": "Russian",
    "영어": "English",
    "일본어": "Japanese",
    "중국어": "Chinese",
    "베트남어": "Vietnamese",
    "프랑스어": "French",
    "스페인어": "Spanish",
}

selected_language = st.selectbox(
    "번역할 언어를 선택하세요",
    [
        "러시아어",
        "영어",
        "일본어",
        "중국어",
        "베트남어",
        "프랑스어",
        "스페인어",
    ],
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


def pil_to_jpeg_data_url(image, quality=92):
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


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


def deduplicate_boxes(lines):
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

            if count_hangul(line.get("text", "")) > count_hangul(existing.get("text", "")):
                result[duplicate_index] = line
            elif line.get("confidence", 0) > existing.get("confidence", 0):
                result[duplicate_index] = line

    return result


def upscale_for_ocr(image, scale=2):
    if scale == 1:
        return image

    w, h = image.size

    return image.resize(
        (max(1, int(w * scale)), max(1, int(h * scale))),
        Image.Resampling.LANCZOS,
    )


def run_korean_candidate_ocr(image):
    """
    원본 + 2배 확대 OCR로 작은 글자를 최대한 후보에 포함합니다.
    이 단계에서는 작은 글자를 크기만으로 제거하지 않습니다.
    """
    variants = [
        (image, 1),
        (upscale_for_ocr(image, 2), 2),
    ]

    all_lines = []

    for variant_image, scale in variants:
        data = pytesseract.image_to_data(
            variant_image,
            lang="kor+eng",
            config="--oem 1 --psm 11",
            output_type=Output.DICT,
        )

        groups = {}

        for i, raw in enumerate(data["text"]):
            raw_text = raw.strip()

            try:
                conf = float(data["conf"][i])
            except (ValueError, TypeError):
                conf = -1

            if not raw_text or conf < 8:
                continue

            key = (
                data["block_num"][i],
                data["par_num"][i],
                data["line_num"][i],
            )

            x = int(data["left"][i] / scale)
            y = int(data["top"][i] / scale)
            w = max(1, int(data["width"][i] / scale))
            h = max(1, int(data["height"][i] / scale))

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

        for group in groups.values():
            line_text = re.sub(
                r"\s+",
                " ",
                " ".join(group["words"]).strip(),
            )

            if not line_text:
                continue

            hangul_count = count_hangul(line_text)
            latin_count = count_latin(line_text)

            if hangul_count == 0 and latin_count == 0:
                continue

            avg_conf = (
                sum(group["conf_values"]) / len(group["conf_values"])
                if group["conf_values"]
                else 0
            )

            all_lines.append(
                {
                    "text": line_text,
                    "x": group["x1"],
                    "y": group["y1"],
                    "w": max(1, group["x2"] - group["x1"]),
                    "h": max(1, group["y2"] - group["y1"]),
                    "confidence": round(avg_conf, 1),
                }
            )

    all_lines = deduplicate_boxes(all_lines)
    all_lines.sort(key=lambda item: (item["y"], item["x"]))
    return all_lines


def cluster_line_regions(candidates, image):
    """
    같은 줄의 OCR 조각을 묶습니다.

    중요:
    좌측/우측 2단 레이아웃이 같은 높이에 있어도
    서로 다른 컬럼이면 하나로 합치지 않습니다.
    """
    if not candidates:
        return []

    cleaned = [item.copy() for item in candidates]
    cleaned.sort(
        key=lambda x: (
            x["y"] + x["h"] / 2,
            x["x"],
        )
    )

    groups = []
    page_center = image.width / 2

    def clearly_different_columns(group, item):
        gx1 = min(x["x"] for x in group)
        gx2 = max(x["x"] + x["w"] for x in group)
        ix1 = item["x"]
        ix2 = item["x"] + item["w"]

        left_right = (
            gx2 < page_center * 0.96
            and ix1 > page_center * 1.04
        )

        right_left = (
            ix2 < page_center * 0.96
            and gx1 > page_center * 1.04
        )

        if left_right or right_left:
            return True

        group_center = (
            sum(x["x"] + x["w"] / 2 for x in group)
            / len(group)
        )

        item_center = item["x"] + item["w"] / 2

        if abs(group_center - item_center) > image.width * 0.35:
            return True

        return False

    for item in cleaned:
        placed = False
        item_cy = item["y"] + item["h"] / 2

        for group in groups:
            if clearly_different_columns(group, item):
                continue

            gy1 = min(x["y"] for x in group)
            gy2 = max(x["y"] + x["h"] for x in group)
            gcy = (gy1 + gy2) / 2
            avg_h = sum(x["h"] for x in group) / len(group)

            y_close = abs(item_cy - gcy) <= max(
                9,
                avg_h * 0.62,
            )

            gx1 = min(x["x"] for x in group)
            gx2 = max(x["x"] + x["w"] for x in group)
            ix1 = item["x"]
            ix2 = item["x"] + item["w"]

            if ix1 > gx2:
                gap = ix1 - gx2
            elif gx1 > ix2:
                gap = gx1 - ix2
            else:
                gap = 0

            max_gap = max(
                16,
                int(avg_h * 1.8),
                int(image.width * 0.020),
            )

            if y_close and gap <= max_gap:
                group.append(item)
                placed = True
                break

        if not placed:
            groups.append([item])

    regions = []

    for group in groups:
        x1 = min(x["x"] for x in group)
        y1 = min(x["y"] for x in group)
        x2 = max(x["x"] + x["w"] for x in group)
        y2 = max(x["y"] + x["h"] for x in group)

        regions.append(
            {
                "x": x1,
                "y": y1,
                "w": x2 - x1,
                "h": y2 - y1,
            }
        )

    regions.sort(key=lambda item: (item["y"], item["x"]))
    return regions


def ocr_line_with_conf(image, lang):
    data = pytesseract.image_to_data(
        image,
        lang=lang,
        config="--oem 1 --psm 7",
        output_type=Output.DICT,
    )

    parts = []
    confs = []

    for i, raw in enumerate(data["text"]):
        item_text = raw.strip()

        if not item_text:
            continue

        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1

        if conf < 0:
            continue

        parts.append(item_text)
        confs.append(conf)

    final_text = re.sub(
        r"\s+",
        " ",
        " ".join(parts),
    ).strip()

    avg_conf = (
        sum(confs) / len(confs)
        if confs
        else 0
    )

    return final_text, avg_conf


def best_korean_line_read(crop):
    """
    원본 / 대비 강화 / 반전 × 원본크기 / 2배 확대를 비교해
    한글이 가장 잘 읽힌 결과를 선택합니다.
    """
    gray = ImageOps.grayscale(crop)

    contrast = ImageOps.autocontrast(
        gray
    ).convert("RGB")

    inverted = ImageOps.invert(
        ImageOps.autocontrast(gray)
    ).convert("RGB")

    base_variants = [
        crop.convert("RGB"),
        contrast,
        inverted,
    ]

    variants = []

    for variant in base_variants:
        variants.append(variant)
        variants.append(
            upscale_for_ocr(variant, 2)
        )

    reads = []

    for variant in variants:
        result_text, conf = ocr_line_with_conf(
            variant,
            "kor+eng",
        )

        h_count = count_hangul(result_text)
        l_count = count_latin(result_text)

        ratio = h_count / max(
            1,
            h_count + l_count,
        )

        score = (
            h_count * 8
            + ratio * 20
            + conf * 0.25
        )

        reads.append(
            (
                score,
                result_text,
                conf,
            )
        )

    reads.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    return reads[0][1], reads[0][2]


def english_ocr_evidence(crop):
    """
    영어 제외 판단은 영어 전용 OCR 결과로만 수행합니다.
    """
    gray = ImageOps.grayscale(crop)

    contrast = ImageOps.autocontrast(
        gray
    ).convert("RGB")

    variants = [
        crop.convert("RGB"),
        contrast,
        upscale_for_ocr(contrast, 2),
    ]

    best = ("", 0.0, 0)

    for variant in variants:
        eng_text, eng_conf = ocr_line_with_conf(
            variant,
            "eng",
        )

        latin_count = count_latin(eng_text)

        score = (
            latin_count * 10
            + eng_conf
        )

        old_score = (
            best[2] * 10
            + best[1]
        )

        if score > old_score:
            best = (
                eng_text,
                eng_conf,
                latin_count,
            )

    return best


def is_probably_english_region(crop, korean_text):
    eng_text, eng_conf, latin_count = english_ocr_evidence(
        crop
    )

    h_count = count_hangul(korean_text)
    k_latin = count_latin(korean_text)

    k_ratio = h_count / max(
        1,
        h_count + k_latin,
    )

    # 한국어가 거의 없고 영어가 또렷한 경우만 자동 제외
    if (
        latin_count >= 4
        and eng_conf >= 58
        and h_count <= 1
    ):
        return True

    # 한국어 OCR 결과 자체도 영문 중심이고 영어 전용 OCR이 강한 경우
    if (
        latin_count >= 5
        and eng_conf >= 65
        and k_ratio < 0.30
    ):
        return True

    return False


def detect_korean_lines(image):
    """
    1차 OCR:
    - 원본 + 2배 확대
    - 반전 이미지 + 2배 확대
    - 작은 글자 크기 필터 없음
    - 좌우 컬럼 병합 방지
    - 확실한 영문만 제거
    """
    candidates = run_korean_candidate_ocr(
        image
    )

    gray = ImageOps.grayscale(image)

    inverted = ImageOps.invert(
        ImageOps.autocontrast(gray)
    ).convert("RGB")

    candidates.extend(
        run_korean_candidate_ocr(inverted)
    )

    candidates = deduplicate_boxes(
        candidates
    )

    regions = cluster_line_regions(
        candidates,
        image,
    )

    final_lines = []

    pad_x = max(
        7,
        int(image.width * 0.006),
    )

    pad_y = max(
        4,
        int(image.height * 0.0025),
    )

    for region in regions:
        x1 = max(
            0,
            region["x"] - pad_x,
        )

        y1 = max(
            0,
            region["y"] - pad_y,
        )

        x2 = min(
            image.width,
            region["x"]
            + region["w"]
            + pad_x,
        )

        y2 = min(
            image.height,
            region["y"]
            + region["h"]
            + pad_y,
        )

        crop = image.crop(
            (
                x1,
                y1,
                x2,
                y2,
            )
        ).convert("RGB")

        korean_text, korean_conf = best_korean_line_read(
            crop
        )

        korean_text = re.sub(
            r"\s+",
            " ",
            korean_text,
        ).strip()

        h_count = count_hangul(
            korean_text
        )

        if h_count == 0:
            continue

        # 영어 전용 OCR 결과로만 영문 오인식 제거
        if is_probably_english_region(
            crop,
            korean_text,
        ):
            continue

        final_lines.append(
            {
                "text": korean_text,
                "x": region["x"],
                "y": region["y"],
                "w": region["w"],
                "h": region["h"],
                "confidence": round(
                    korean_conf,
                    1,
                ),
            }
        )

    final_lines = deduplicate_boxes(
        final_lines
    )

    final_lines.sort(
        key=lambda item: (
            item["y"],
            item["x"],
        )
    )

    return final_lines


def draw_detection_preview(image, lines):
    preview = image.copy()
    draw = ImageDraw.Draw(preview)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            max(
                16,
                int(image.width * 0.022),
            ),
        )
    except Exception:
        font = ImageFont.load_default()

    line_width = max(
        2,
        int(image.width * 0.004),
    )

    for index, line in enumerate(
        lines,
        start=1,
    ):
        x1 = line["x"]
        y1 = line["y"]
        x2 = x1 + line["w"]
        y2 = y1 + line["h"]

        pad = max(
            3,
            int(image.width * 0.004),
        )

        x1 = max(
            0,
            x1 - pad,
        )

        y1 = max(
            0,
            y1 - pad,
        )

        x2 = min(
            image.width,
            x2 + pad,
        )

        y2 = min(
            image.height,
            y2 + pad,
        )

        draw.rectangle(
            [x1, y1, x2, y2],
            outline=(255, 0, 0),
            width=line_width,
        )

        label = str(index)

        label_x = x1

        label_y = max(
            0,
            y1
            - max(
                22,
                int(image.width * 0.03),
            ),
        )

        bbox = draw.textbbox(
            (label_x, label_y),
            label,
            font=font,
        )

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


def clean_json_text(text):
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
        )

        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        )

    return cleaned.strip()


def classify_ocr_regions_with_ai(
    image,
    lines,
):
    """
    OCR 후보를 실제 이미지와 함께 AI가 다시 검수합니다.
    """
    if not lines:
        return [], []

    numbered_preview = draw_detection_preview(
        image,
        lines,
    )

    candidate_text = []

    for idx, line in enumerate(
        lines,
        start=1,
    ):
        candidate_text.append(
            f"{idx}. OCR={line['text']!r}, "
            f"x={line['x']}, y={line['y']}, "
            f"w={line['w']}, h={line['h']}"
        )

    prompt = f"""
당신은 한국 화장품 상세페이지 OCR 검수기입니다.

첫 번째 이미지는 원본 상세페이지입니다.
두 번째 이미지는 OCR 후보 박스에 번호를 표시한 이미지입니다.

아래 각 OCR 후보를 반드시 5가지 중 하나로 분류하세요.

PAGE_COPY
- 실제 상세페이지 위에 직접 배치된 번역 대상 문구
- 제목, 설명문, 사용법, 표 항목명, 표의 한국어 값,
  주의사항, 시험 조건, 각주 등
- 글자가 작아도 페이지 카피라면 반드시 PAGE_COPY

INGREDIENTS_BLOCK
- '전성분'이라는 섹션 제목이 아니라
  실제 전체 성분명이 길게 나열된 본문
- 이 목록은 공식 영문 INCI로 유지할 예정이므로 번역 대상에서 제외
- '전성분' 제목 자체는 PAGE_COPY

EMBEDDED_IMAGE_TEXT
- 제품 패키지, 특허증, 시험성적서, 인증서,
  사진, 문서 이미지 내부에 이미 인쇄된 글자

ENGLISH_OR_NONKOREAN
- 순수 영어/기타 언어
- 단, 'DRY 푸석할 때'처럼 영어+한글이 하나의 디자인 문구라면 PAGE_COPY

NOISE
- 선, 물방울, 아이콘, 그래픽 등을 글자로 오인한 것

추가 규칙:
1. 실제 이미지 위치를 보고 판단하세요.
2. 작은 본문/각주를 크기 때문에 제외하지 마세요.
3. OCR 오타가 있으면 corrected_text에 이미지에 실제로 보이는 문구를 적으세요.
4. 서로 다른 좌/우 컬럼 문장이 한 OCR 박스로 합쳐졌다면
   PAGE_COPY로 두되 reason에 '서로 다른 문장이 합쳐짐'이라고 적으세요.
5. 모든 번호를 빠짐없이 반환하세요.

OCR 후보:
{chr(10).join(candidate_text)}

JSON만 반환하세요.

형식:
{{
  "items": [
    {{
      "id": 1,
      "class": "PAGE_COPY",
      "corrected_text": "이미지에 보이는 정확한 문구",
      "reason": "상세페이지 제목"
    }}
  ]
}}
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
                        "image_url": pil_to_jpeg_data_url(
                            image
                        ),
                        "detail": "high",
                    },
                    {
                        "type": "input_image",
                        "image_url": pil_to_jpeg_data_url(
                            numbered_preview
                        ),
                        "detail": "high",
                    },
                ],
            }
        ],
    )

    try:
        parsed = json.loads(
            clean_json_text(
                response.output_text
            )
        )

        returned_items = parsed.get(
            "items",
            [],
        )

    except Exception:
        fallback = []

        for idx, line in enumerate(
            lines,
            start=1,
        ):
            fallback.append(
                {
                    **line,
                    "source_id": idx,
                    "class": "PAGE_COPY",
                    "reason": "AI JSON 파싱 실패 - 수동 검수 필요",
                }
            )

        return fallback, []

    valid_classes = {
        "PAGE_COPY",
        "INGREDIENTS_BLOCK",
        "EMBEDDED_IMAGE_TEXT",
        "ENGLISH_OR_NONKOREAN",
        "NOISE",
    }

    by_id = {}

    for item in returned_items:
        try:
            item_id = int(
                item.get("id")
            )
        except Exception:
            continue

        item_class = str(
            item.get(
                "class",
                "",
            )
        ).strip().upper()

        if item_class not in valid_classes:
            item_class = "PAGE_COPY"

        corrected_text = str(
            item.get(
                "corrected_text",
                "",
            )
        ).strip()

        reason = str(
            item.get(
                "reason",
                "",
            )
        ).strip()

        by_id[item_id] = {
            "class": item_class,
            "corrected_text": corrected_text,
            "reason": reason,
        }

    kept = []
    excluded = []

    for idx, line in enumerate(
        lines,
        start=1,
    ):
        judgement = by_id.get(
            idx,
            {
                "class": "PAGE_COPY",
                "corrected_text": line["text"],
                "reason": "AI 응답에서 누락되어 안전하게 유지",
            },
        )

        result = {
            **line,
            "source_id": idx,
            "class": judgement["class"],
            "reason": judgement["reason"],
        }

        corrected_text = judgement.get(
            "corrected_text",
            "",
        ).strip()

        if corrected_text:
            result["ocr_text"] = line["text"]
            result["text"] = corrected_text

        if result["class"] == "PAGE_COPY":
            kept.append(result)
        else:
            excluded.append(result)

    return kept, excluded


if "all_results" not in st.session_state:
    st.session_state.all_results = []

if "ocr_results" not in st.session_state:
    st.session_state.ocr_results = {}


if uploaded_files:
    st.subheader("업로드된 이미지")

    st.write(
        f"총 {len(uploaded_files)}장의 이미지가 업로드되었습니다."
    )

    st.info(
        "업로드한 순서대로 번역됩니다. (1번 → 2번 → 3번...)"
    )

    for idx, file in enumerate(
        uploaded_files,
        start=1,
    ):
        st.markdown(
            f"**{idx}번 이미지: {file.name}**"
        )

    if st.button(
        "🚀 AI 번역 시작",
        type="primary",
    ):
        all_results = []
        progress = st.progress(0)

        for idx, uploaded_file in enumerate(
            uploaded_files,
            start=1,
        ):
            with st.spinner(
                f"{idx}/{len(uploaded_files)} 번역 중..."
            ):
                base64_image, preview_image = image_to_base64(
                    uploaded_file
                )

                target_language = language_map[
                    selected_language
                ]

                prompt = f"""
이 이미지는 한국 화장품 상세페이지의 분할 이미지입니다.

1. 이미지 안의 한국어를 위에서 아래 순서대로 읽으세요.
2. 각 문구를 {target_language}로 자연스럽게 번역하세요.

출력 형식:

[한국어]
원문

[{selected_language}]
번역문

규칙:
1. 이미지에 실제로 보이는 한국어만 적으세요.
2. 숫자, %, ppm, ml, g, 날짜, 시험 수치는 원문 그대로 유지하세요.
3. 브랜드명과 제품명은 함부로 번역하지 마세요.
4. 실제 전성분 목록은 번역하지 말고:
   [전성분]
   영문 INCI 유지
   라고 표시하세요.
5. 화장품 광고 문구는 치료/완치/의약품 효능처럼 과장하지 마세요.
6. 원문 의미를 최대한 유지하세요.
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

                progress.progress(
                    idx / len(uploaded_files)
                )

        st.session_state.all_results = all_results
        st.session_state.ocr_results = {}

        st.success(
            "✅ 모든 이미지 번역이 완료되었습니다."
        )


if st.session_state.all_results:
    st.markdown("---")
    st.header("📑 이미지별 번역 결과")

    combined_text = ""

    for item in st.session_state.all_results:
        st.markdown("---")

        st.subheader(
            f"{item['index']}번 이미지 · {item['file_name']}"
        )

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
                f"🔎 {item['index']}번 OCR + AI 검수",
                key=f"detect_{item['index']}",
                use_container_width=True,
            )

        if find_boxes:
            with st.spinner(
                f"{item['index']}번 이미지 OCR 및 AI 검수 중..."
            ):
                try:
                    raw_lines = detect_korean_lines(
                        item["image"]
                    )

                    raw_preview = draw_detection_preview(
                        item["image"],
                        raw_lines,
                    )

                    kept_lines, excluded_lines = classify_ocr_regions_with_ai(
                        item["image"],
                        raw_lines,
                    )

                    final_preview = draw_detection_preview(
                        item["image"],
                        kept_lines,
                    )

                    st.session_state.ocr_results[
                        item["index"]
                    ] = {
                        "raw_lines": raw_lines,
                        "raw_preview": raw_preview,
                        "lines": kept_lines,
                        "excluded": excluded_lines,
                        "preview": final_preview,
                    }

                    st.success(
                        f"OCR 후보 {len(raw_lines)}개 → "
                        f"최종 번역 대상 {len(kept_lines)}개 / "
                        f"제외 {len(excluded_lines)}개"
                    )

                except Exception as exc:
                    st.error(
                        "OCR 또는 AI 검수 중 오류가 발생했습니다."
                    )
                    st.code(str(exc))

        col_image, col_text = st.columns(
            [1, 1],
            gap="large",
        )

        with col_image:
            st.markdown("### 🖼️ 원본 이미지")

            st.image(
                item["image"],
                use_container_width=True,
            )

        with col_text:
            st.markdown(
                f"### 🌐 {item['language']} 번역"
            )

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
            ocr_data = st.session_state.ocr_results[
                item["index"]
            ]

            raw_lines = ocr_data.get(
                "raw_lines",
                [],
            )

            raw_preview = ocr_data.get(
                "raw_preview"
            )

            final_lines = ocr_data.get(
                "lines",
                [],
            )

            excluded = ocr_data.get(
                "excluded",
                [],
            )

            preview = ocr_data.get(
                "preview"
            )

            st.markdown(
                "### 🔎 OCR + AI 검수 결과"
            )

            tab_final, tab_raw, tab_excluded = st.tabs(
                [
                    "✅ 최종 번역 대상",
                    "🔍 OCR 원본 후보",
                    "🚫 제외된 영역",
                ]
            )

            with tab_final:
                if preview is not None:
                    st.image(
                        preview,
                        caption="AI 검수 후 최종 번역 대상",
                        use_container_width=True,
                    )

                    st.download_button(
                        label="📥 최종 위치 검사 PNG",
                        data=pil_to_png_bytes(
                            preview
                        ),
                        file_name=f"{item['index']:02d}_final_ocr_preview.png",
                        mime="image/png",
                        key=f"download_final_preview_{item['index']}",
                        use_container_width=True,
                    )

                if final_lines:
                    for line_index, line in enumerate(
                        final_lines,
                        start=1,
                    ):
                        st.markdown(
                            f"**{line_index}. {line['text']}**  \n"
                            f"위치: x={line['x']}, y={line['y']}  \n"
                            f"크기: {line['w']}×{line['h']}px  \n"
                            f"이유: {line.get('reason', '')}"
                        )

            with tab_raw:
                if raw_preview is not None:
                    st.image(
                        raw_preview,
                        caption=f"OCR 원본 후보 {len(raw_lines)}개",
                        use_container_width=True,
                    )

                for raw_index, raw_line in enumerate(
                    raw_lines,
                    start=1,
                ):
                    st.markdown(
                        f"**{raw_index}. {raw_line['text']}** "
                        f"(x={raw_line['x']}, y={raw_line['y']}, "
                        f"{raw_line['w']}×{raw_line['h']}px)"
                    )

            with tab_excluded:
                if excluded:
                    for line in excluded:
                        st.markdown(
                            f"**원본 번호 {line.get('source_id')}. "
                            f"{line['text']}**  \n"
                            f"분류: `{line.get('class')}`  \n"
                            f"이유: {line.get('reason', '')}"
                        )
                        st.markdown("---")
                else:
                    st.info(
                        "AI가 제외한 영역이 없습니다."
                    )

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
