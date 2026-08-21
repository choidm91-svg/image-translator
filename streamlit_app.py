import base64
import io
import json
import re
from typing import Dict, List

import streamlit as st
from openai import OpenAI
from PIL import Image

st.set_page_config(
    page_title="AI 상세페이지 번역기 v11.7",
    page_icon="🌐",
    layout="wide",
)

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ------------------------------------------------------------
# 시장 / 언어 프리셋
# ------------------------------------------------------------
MARKET_LANGUAGE = {
    "러시아": "Russian",
    "미국": "American English",
    "영국": "British English",
    "글로벌 영어권": "International English",
    "일본": "Japanese",
    "중국 본토": "Simplified Chinese",
    "대만": "Traditional Chinese",
    "베트남": "Vietnamese",
    "프랑스": "French",
    "스페인": "Spanish (Spain)",
    "멕시코": "Spanish (Mexico)",
    "라틴아메리카": "Neutral Latin American Spanish",
}

MARKET_DISPLAY_LANGUAGE = {
    "러시아": "러시아어",
    "미국": "영어",
    "영국": "영어",
    "글로벌 영어권": "영어",
    "일본": "일본어",
    "중국 본토": "중국어(간체)",
    "대만": "중국어(번체)",
    "베트남": "베트남어",
    "프랑스": "프랑스어",
    "스페인": "스페인어",
    "멕시코": "스페인어",
    "라틴아메리카": "스페인어",
}

MARKET_RULES = {
    "러시아": """
- 러시아 화장품 상세페이지에서 자연스럽게 쓰는 간결한 표현을 사용하세요.
- 절대적 안전 보장(완전히 안전, 자극 없음, 누구나 안심)을 만들지 마세요.
- 시험 결과는 시험 범위 안에서만 표현하세요.
- 치료·치유·의학적 재생처럼 의약품으로 오인될 수 있는 표현은 보습, 컨디셔닝, 외관 개선 중심으로 순화하세요.
""",
    "미국": """
- 미국 화장품 카피처럼 짧고 자연스럽게 작성하세요.
- 질환 치료/예방, 신체 구조·기능 변화처럼 보이는 drug claim은 피하세요.
- cure, treat, heal, regenerate, 100% safe 같은 절대 표현을 사용하지 마세요.
- appearance, feel, helps, visibly 같은 화장품 범위 표현을 우선하세요.
""",
    "영국": """
- 자연스러운 British English 화장품 카피로 작성하세요.
- medicinal/therapeutic claim처럼 보일 수 있는 치료·치유·질환 예방 표현을 피하세요.
- 외관, 사용감, 보습 중심으로 보수적으로 표현하세요.
""",
    "글로벌 영어권": """
- 자연스러운 International English를 사용하세요.
- 치료·치유·질환 예방·절대 안전 보장 표현을 피하세요.
- helps, visibly, appearance of, feels 같은 화장품 범위 표현을 우선하세요.
""",
    "일본": """
- 일본 화장품 상세페이지에서 자연스러운 짧은 카피로 현지화하세요.
- 의약부외품 근거가 없는 한 치료·치유·재생을 단정하지 마세요.
- 肌を整える, うるおいを与える, ハリ感, 乾燥を防ぐ 같은 화장품 범위 표현을 우선하세요.
""",
    "중국 본토": """
- 중국 본토 소비자에게 자연스러운 간체 중국어 화장품 카피를 사용하세요.
- 第一, 最, 100%有效, 绝对安全, 零刺激 같은 절대적·최상급 표현을 피하세요.
- 치료·의학적 효능보다 보습, 컨디셔닝, 외관 개선 중심으로 표현하세요.
""",
    "대만": """
- 대만 화장품 상세페이지에 자연스러운 번체 중국어로 현지화하세요.
- 치료, 치유, 의학적 재생을 단정하지 말고 保濕、調理肌膚、維持肌膚狀態、改善外觀 중심으로 표현하세요.
- 절대적 안전·효능 보장을 피하세요.
""",
    "베트남": """
- 베트남 화장품 상세페이지에 맞는 짧고 자연스러운 문장으로 현지화하세요.
- 치료·치유·염증 개선·의학적 재생을 단정하지 마세요.
- an toàn tuyệt đối, không gây kích ứng 100% 같은 절대적 안전 보장을 피하세요.
""",
    "프랑스": """
- 프랑스 화장품 상세페이지처럼 짧고 자연스럽게 작성하세요.
- médicament처럼 보일 수 있는 치료·치유·질환 예방 표현을 피하세요.
- hydratation, confort, apparence, peau visiblement plus lisse 같은 화장품 범위 표현을 우선하세요.
""",
    "스페인": """
- 스페인 소비자에게 자연스러운 유럽 스페인어 화장품 카피로 현지화하세요.
- 치료·치유·질환 예방 표현을 피하고 hidratación, confort, apariencia 중심으로 표현하세요.
""",
    "멕시코": """
- 멕시코 화장품 상세페이지에 자연스러운 스페인어를 사용하세요.
- 치료·치유·의학적 재생을 단정하지 마세요.
- 100% seguro, sin irritación, cura 같은 절대적·치료성 표현을 피하세요.
""",
    "라틴아메리카": """
- 중립적인 라틴아메리카 스페인어를 사용하세요.
- 치료·치유·질환 예방처럼 의약품으로 오인될 수 있는 표현을 피하세요.
- hidratación, cuidado, confort, apariencia de la piel 중심으로 표현하세요.
""",
}

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.title("🌐 AI 상세페이지 번역기 v11.7")
st.caption(
    "원클릭 실무 모드: 국가 선택 → 번역 → 검수 필요한 문장만 확인 → 전체 다운로드"
)
st.success("✅ v11.7 AUTO-SPLIT · NO-OCR · 긴 상세페이지는 자동 분할해서 번역합니다")

selected_market = st.selectbox(
    "판매 국가 / 시장",
    list(MARKET_LANGUAGE.keys()),
    index=0,
)

target_language = MARKET_LANGUAGE[selected_market]
display_language = MARKET_DISPLAY_LANGUAGE[selected_market]

with st.expander("고급 설정", expanded=False):
    safety_mode = st.checkbox(
        "현지 화장품 광고 표현을 보수적으로 순화",
        value=True,
    )
    short_mode = st.checkbox(
        "디자인용 짧은 번역",
        value=True,
    )
    st.caption("기본값 그대로 사용해도 됩니다.")

st.info(
    f"현재 프리셋: **{selected_market} / {display_language}** · 광고 표현 순화 + 디자인용 짧은 번역"
)

# 고정 번역 사전
if "translation_dictionary" not in st.session_state:
    st.session_state.translation_dictionary = {}

with st.expander("📘 고정 번역 사전", expanded=False):
    st.caption(
        "한 번 확정한 표현을 계속 같은 번역으로 사용합니다. "
        "형식: 한국어 = 번역문 (한 줄에 하나)"
    )
    default_dict_text = "\n".join(
        f"{k} = {v}" for k, v in st.session_state.translation_dictionary.items()
    )
    dictionary_text = st.text_area(
        "고정 번역",
        value=default_dict_text,
        height=160,
        placeholder="피부 저자극 테스트 완료 = Пройден тест на низкий потенциал раздражения кожи\n전성분 = 영문 INCI 유지",
        key="dictionary_editor",
    )

    parsed_dictionary = {}
    for line in dictionary_text.splitlines():
        if "=" not in line:
            continue
        source, target = line.split("=", 1)
        source = source.strip()
        target = target.strip()
        if source and target:
            parsed_dictionary[source] = target
    st.session_state.translation_dictionary = parsed_dictionary

    dict_json = json.dumps(parsed_dictionary, ensure_ascii=False, indent=2)
    st.download_button(
        "📥 번역 사전 JSON 다운로드",
        data=dict_json,
        file_name=f"translation_dictionary_{selected_market}.json",
        mime="application/json",
        use_container_width=True,
    )

uploaded_files = st.file_uploader(
    "상세페이지 JPG / PNG 이미지를 여러 장 업로드하세요",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def image_to_data_url(uploaded_file):
    uploaded_file.seek(0)
    image = Image.open(uploaded_file).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    uploaded_file.seek(0)
    return f"data:image/jpeg;base64,{encoded}", image




def pil_to_data_url(image: Image.Image):
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=94)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def split_tall_image(image: Image.Image, max_height: int = 6000, overlap: int = 450):
    """
    세로로 매우 긴 상세페이지를 자동 분할합니다.
    각 조각은 450px 정도 겹치게 잘라 경계에 걸린 문장 누락을 줄입니다.
    """
    width, height = image.size

    if height <= max_height:
        return [
            {
                "index": 1,
                "y1": 0,
                "y2": height,
                "image": image,
            }
        ]

    chunks = []
    y1 = 0
    index = 1

    while y1 < height:
        y2 = min(height, y1 + max_height)
        crop = image.crop((0, y1, width, y2)).convert("RGB")
        chunks.append(
            {
                "index": index,
                "y1": y1,
                "y2": y2,
                "image": crop,
            }
        )

        if y2 >= height:
            break

        y1 = max(0, y2 - overlap)
        index += 1

    return chunks


def merge_chunk_segments(segment_groups):
    """
    자동 분할 조각의 겹치는 영역에서 동일 문구가 두 번 잡히는 것을 제거합니다.
    조각 순서를 그대로 유지합니다.
    """
    merged = []
    seen = set()

    for group in segment_groups:
        for seg in group:
            key = normalize_key(seg.get("korean", ""))
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(dict(seg))

    for idx, seg in enumerate(merged, start=1):
        seg["order"] = idx

    return merged

def clean_json_text(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_json_response(text):
    cleaned = clean_json_text(text)
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    match = re.search(r"\{.*\}", cleaned, re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def normalize_key(text):
    return re.sub(r"[^가-힣A-Za-z0-9]", "", (text or "").lower())


def extract_numbers(text):
    return re.findall(r"\d+(?:[.,]\d+)?%?|\d+(?:[.,]\d+)?", text or "")


def visual_units(text):
    """디자인 길이 비교용 간단한 시각 폭 추정치."""
    units = 0.0
    for ch in text or "":
        if ch == "\n":
            continue
        if re.match(r"[가-힣一-龥ぁ-んァ-ン]", ch):
            units += 1.0
        elif ch.isspace():
            units += 0.35
        elif ch.isdigit():
            units += 0.55
        else:
            units += 0.55
    return max(1.0, units)


def length_check(korean, translation, seg_type):
    source = visual_units(korean)
    target = visual_units(translation)
    ratio = target / source

    if seg_type in {"headline", "label"}:
        limit = 1.35
    elif seg_type in {"footnote", "test_value"}:
        limit = 1.60
    else:
        limit = 1.50

    return {
        "ratio": round(ratio, 2),
        "limit": limit,
        "too_long": ratio > limit,
    }


def apply_dictionary(korean, translation, dictionary):
    # 완전 일치 우선
    if korean in dictionary:
        return dictionary[korean], True

    # 짧은 고정 용어가 문장에 포함되어 있으면 번역문 전체를 억지 치환하지 않음.
    return translation, False


def deduplicate_segments(segments):
    seen = set()
    result = []
    for seg in segments:
        key = normalize_key(seg.get("korean", ""))
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(seg)
    for idx, seg in enumerate(result, start=1):
        seg["order"] = idx
    return result


def build_dictionary_prompt(dictionary):
    if not dictionary:
        return "고정 번역 사전 없음"
    lines = [f"- {k} => {v}" for k, v in dictionary.items()]
    return "\n".join(lines)


def is_meta_commentary_segment(korean, translation):
    """AI가 이미지 속 문구가 아니라 해상도/판독 안내문을 만들어낸 경우 제거합니다."""
    combined = f"{korean} {translation}".lower()
    patterns = [
        "이미지 해상도",
        "해상도가 너무 낮",
        "문자 식별이 어렵",
        "고해상도 이미지",
        "텍스트가 선명",
        "이미지를 다시",
        "업로드해",
        "resolution is too low",
        "low resolution",
        "upload a higher",
    ]
    return any(pattern.lower() in combined for pattern in patterns)


def scrub_false_resolution_reason(reason, image_width):
    """가로 800px 이상이면 AI가 추측한 '저해상도' 검수 사유를 제거합니다."""
    reason = str(reason or "").strip()
    if image_width < 800:
        return reason

    low_res_terms = [
        "해상도",
        "저해상도",
        "고해상도",
        "resolution",
    ]
    if any(term.lower() in reason.lower() for term in low_res_terms):
        return ""
    return reason


def translate_detail_image(image_url, image_width, image_height, target_language, target_market, dictionary, safety_mode, short_mode):
    safety_instruction = (
        "치료, 완치, 재생, 의약품 효능처럼 보이는 표현은 원문 의미를 유지하는 범위에서만 보수적으로 순화하세요. "
        "시험 결과를 일반적인 안전성 보장이나 의학적 효능으로 확대하지 마세요."
        if safety_mode
        else "원문 의미를 최대한 그대로 유지하세요."
    )

    concise_instruction = (
        """
디자인용 길이 규칙:
- headline / label: 최대한 한 줄, 짧게.
- body: 핵심 정보만 유지해 1~2개의 짧은 문장.
- footnote / test_value: 가장 짧고 명확하게.
- 번역문이 한국어보다 지나치게 길어지지 않게 하세요.
- 긴 경우 의미를 보존하면서 더 짧은 현지 표현을 우선하세요.
"""
        if short_mode
        else ""
    )

    dictionary_prompt = build_dictionary_prompt(dictionary)

    prompt = f"""
이 이미지는 한국 화장품 상세페이지입니다.
실제 업로드 이미지 크기는 가로 {image_width}px × 세로 {image_height}px 입니다.
이미지 크기는 이미 앱이 직접 측정했으므로 해상도를 추측하지 마세요.
가로가 800px 이상이면 절대로 "해상도가 낮다"거나 "고해상도 이미지를 다시 업로드하라"는 안내를 만들지 마세요.
판독이 어려운 일부 문구가 있더라도 읽을 수 있는 한국어만 추출하고, 이미지 품질에 대한 설명은 출력하지 마세요.

이미지 전체를 위에서 아래, 왼쪽에서 오른쪽 순서로 읽고
실제로 보이는 한국어만 추출한 뒤 {target_market} 시장용 {target_language}로 현지화하세요.

반드시 JSON만 반환하세요.

형식:
{{
  "segments": [
    {{
      "order": 1,
      "korean": "한국어 원문",
      "translation": "현지화 번역",
      "type": "headline | body | label | footnote | test_value | ingredients | other",
      "review_required": false,
      "review_reason": ""
    }}
  ]
}}

규칙:
1. 실제로 보이는 한국어만 적으세요.
2. 작은 글자, 각주, 사용 전/사용 후, 표 안의 한국어, 테스트 설명도 확인하세요.
3. 영어만 있는 브랜드명/제품명/패키지 영문은 새로 번역하지 마세요.
4. 숫자, %, ppm, ml, g, 날짜, 시험 수치, 별표(*)는 그대로 유지하세요.
5. 실제 전체 전성분 목록은 번역하지 마세요. 전체 전성분이 보이면 korean="전성분", translation="영문 INCI 유지", type="ingredients" 로 한 번만 적으세요.
6. 제품 패키지, 시험성적서 이미지, 도장/스탬프, 인증서, QR코드 안 문구는 상세페이지 본문 카피가 아니라면 제외하세요.
7. 같은 문구를 중복해서 적지 마세요.
8. 여러 줄로 나뉜 하나의 문장은 합쳐서 적으세요.
9. 단순 직역이 아니라 {target_market} 소비자가 자연스럽게 읽는 화장품 카피로 현지화하세요.
10. 원문보다 강한 효능·안전성 주장을 새로 만들지 마세요.
11. {safety_instruction}
12. 시장별 규칙:\n{MARKET_RULES.get(target_market, '')}
13. {concise_instruction}
14. 고정 번역 사전은 반드시 우선 적용하세요:\n{dictionary_prompt}
15. review_required=true 로 표시할 조건:
   - 원문 의미가 모호해 두 가지 이상 번역 가능
   - 광고 규제상 표현을 크게 순화해야 해서 원문과 차이가 커짐
   - 숫자/시험 결과/기간/퍼센트가 복잡해 사람이 확인하는 것이 안전함
   - 번역문이 디자인에 넣기에는 길어질 가능성이 큼
   - 특정 문구 자체가 흐리거나 가려져 정확한 판독이 불확실함
16. 이미지 전체의 해상도에 대해 추측하거나 경고하지 마세요. 실제 크기는 위에 제공된 값을 따르세요.
17. 이미지에 한국어가 전혀 없다면 {"segments": []} 만 반환하세요.
18. 위 조건이 없으면 review_required=false.
19. 설명/해설 없이 JSON만 반환하세요.
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

        # 이미지에 없는 '저해상도 안내문' 같은 메타 응답은 결과에서 제거
        if is_meta_commentary_segment(korean, translation):
            continue

        translation, dict_applied = apply_dictionary(korean, translation, dictionary)
        length_info = length_check(korean, translation, seg_type)

        ai_review = bool(seg.get("review_required", False))
        reasons = []
        ai_reason = scrub_false_resolution_reason(
            seg.get("review_reason", ""),
            image_width,
        )
        if ai_review and ai_reason:
            reasons.append(ai_reason)

        source_numbers = extract_numbers(korean)
        target_numbers = extract_numbers(translation)
        if source_numbers != target_numbers and seg_type != "ingredients":
            reasons.append("숫자/시험값 확인 필요")

        if length_info["too_long"]:
            reasons.append(
                f"디자인 길이 초과 ({length_info['ratio']}× / 권장 {length_info['limit']}× 이하)"
            )

        if dict_applied:
            dictionary_status = "고정 사전 적용"
        else:
            dictionary_status = ""

        cleaned.append(
            {
                "order": idx,
                "korean": korean,
                "translation": translation,
                "type": seg_type,
                "review_required": bool(reasons),
                "review_reason": " / ".join(dict.fromkeys(reasons)),
                "length_ratio": length_info["ratio"],
                "length_limit": length_info["limit"],
                "too_long": length_info["too_long"],
                "dictionary_status": dictionary_status,
            }
        )

    cleaned = deduplicate_segments(cleaned)
    return cleaned, response.output_text


def translate_detail_image_auto_split(
    image,
    target_language,
    target_market,
    dictionary,
    safety_mode,
    short_mode,
):
    """
    긴 상세페이지는 자동으로 세로 분할해서 순서대로 번역합니다.
    Tesseract/OCR 패키지를 전혀 사용하지 않습니다.
    """
    chunks = split_tall_image(image, max_height=6000, overlap=450)
    segment_groups = []
    raw_responses = []

    for chunk in chunks:
        chunk_image = chunk["image"]
        chunk_url = pil_to_data_url(chunk_image)

        segments, raw_response = translate_detail_image(
            chunk_url,
            chunk_image.width,
            chunk_image.height,
            target_language,
            target_market,
            dictionary,
            safety_mode,
            short_mode,
        )

        # 어떤 조각에서 나온 문구인지 내부적으로 보존
        for seg in segments:
            seg["chunk_index"] = chunk["index"]
            seg["chunk_y1"] = chunk["y1"]
            seg["chunk_y2"] = chunk["y2"]

        segment_groups.append(segments)
        raw_responses.append(raw_response)

    merged = merge_chunk_segments(segment_groups)

    return merged, "\n\n--- CHUNK ---\n\n".join(raw_responses), len(chunks)


def shorten_translations(segments, target_market, target_language):
    candidates = [
        {
            "order": seg["order"],
            "korean": seg["korean"],
            "translation": seg["translation"],
            "type": seg["type"],
        }
        for seg in segments
        if seg.get("too_long")
    ]

    if not candidates:
        return segments

    prompt = f"""
다음 {target_market} 시장용 {target_language} 화장품 번역문 중 길이가 긴 문장만 더 짧게 줄이세요.
핵심 의미, 숫자, 시험값, 광고 리스크 완화 수준은 그대로 유지하세요.
headline/label은 가능한 한 한 줄, body는 1~2개의 짧은 문장으로 만드세요.

반드시 JSON만 반환:
{{"items":[{{"order":1,"translation":"짧은 번역"}}]}}

입력:
{json.dumps(candidates, ensure_ascii=False)}
"""

    response = client.responses.create(
        model="gpt-5",
        input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
    )
    parsed = parse_json_response(response.output_text)
    mapping = {}
    if isinstance(parsed, dict):
        for item in parsed.get("items", []) or []:
            try:
                order = int(item.get("order"))
            except Exception:
                continue
            mapping[order] = str(item.get("translation", "")).strip()

    updated = []
    for seg in segments:
        new_seg = dict(seg)
        if seg["order"] in mapping and mapping[seg["order"]]:
            new_seg["translation"] = mapping[seg["order"]]
            length_info = length_check(new_seg["korean"], new_seg["translation"], new_seg["type"])
            new_seg["length_ratio"] = length_info["ratio"]
            new_seg["length_limit"] = length_info["limit"]
            new_seg["too_long"] = length_info["too_long"]
            if not new_seg["too_long"]:
                reasons = [
                    reason.strip()
                    for reason in new_seg.get("review_reason", "").split("/")
                    if reason.strip() and "디자인 길이 초과" not in reason
                ]
                new_seg["review_reason"] = " / ".join(reasons)
                new_seg["review_required"] = bool(reasons)
        updated.append(new_seg)
    return updated


def segments_to_text(file_name, market, language, segments):
    parts = [
        f"===== {file_name} =====",
        f"판매 시장: {market}",
        f"번역 언어: {language}",
        "",
    ]
    for seg in segments:
        parts.extend(
            [
                f"[{seg['order']}]",
                "[한국어]",
                seg["korean"],
                "",
                f"[{language}]",
                seg["translation"],
                "",
            ]
        )
    return "\n".join(parts)

# ------------------------------------------------------------
# Run translation
# ------------------------------------------------------------
if "v117_results" not in st.session_state:
    st.session_state.v117_results = []

if uploaded_files:
    if st.button("🚀 원클릭 전체 번역", type="primary", use_container_width=True):
        results = []
        progress = st.progress(0)

        for idx, uploaded_file in enumerate(uploaded_files, start=1):
            with st.spinner(f"{idx}/{len(uploaded_files)} · {uploaded_file.name} 번역 중..."):
                image_url, image = image_to_data_url(uploaded_file)
                try:
                    segments, raw_response, chunk_count = translate_detail_image_auto_split(
                        image,
                        target_language,
                        selected_market,
                        st.session_state.translation_dictionary,
                        safety_mode,
                        short_mode,
                    )
                    results.append(
                        {
                            "index": idx,
                            "file_name": uploaded_file.name,
                            "image": image,
                            "segments": segments,
                            "raw_response": raw_response,
                            "chunk_count": chunk_count,
                        }
                    )
                except Exception as exc:
                    results.append(
                        {
                            "index": idx,
                            "file_name": uploaded_file.name,
                            "image": image,
                            "segments": [],
                            "raw_response": "",
                            "chunk_count": 0,
                            "error": str(exc),
                        }
                    )
            progress.progress(idx / len(uploaded_files))

        st.session_state.v117_results = results
        st.success("✅ 번역 완료")

# ------------------------------------------------------------
# Results
# ------------------------------------------------------------
if st.session_state.v117_results:
    total_segments = sum(len(item.get("segments", [])) for item in st.session_state.v117_results)
    total_review = sum(
        1
        for item in st.session_state.v117_results
        for seg in item.get("segments", [])
        if seg.get("review_required")
    )
    total_long = sum(
        1
        for item in st.session_state.v117_results
        for seg in item.get("segments", [])
        if seg.get("too_long")
    )

    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("전체 문장", total_segments)
    m2.metric("⚠ 검수 필요", total_review)
    m3.metric("✂ 길이 초과", total_long)

    if total_long > 0:
        if st.button("✂️ 긴 문장만 자동 축약", use_container_width=True):
            with st.spinner("길이가 긴 번역만 다시 짧게 줄이고 있습니다..."):
                updated_results = []
                for item in st.session_state.v117_results:
                    new_item = dict(item)
                    if not item.get("error"):
                        new_item["segments"] = shorten_translations(
                            item.get("segments", []),
                            selected_market,
                            target_language,
                        )
                    updated_results.append(new_item)
                st.session_state.v117_results = updated_results
            st.rerun()

    tab_review, tab_all = st.tabs(["⚠ 검수 필요한 문장만", "📑 전체 번역"])

    with tab_review:
        review_found = False
        for item in st.session_state.v117_results:
            review_segments = [seg for seg in item.get("segments", []) if seg.get("review_required")]
            if not review_segments:
                continue
            review_found = True
            st.subheader(f"{item['index']}번 · {item['file_name']}")
            for seg in review_segments:
                st.markdown(f"**{seg['order']}. {seg['type']}**")
                st.write(f"한국어: {seg['korean']}")
                st.write(f"번역: {seg['translation']}")
                if seg.get("review_reason"):
                    st.warning(seg["review_reason"])
                if seg.get("dictionary_status"):
                    st.caption(seg["dictionary_status"])
                st.markdown("---")
        if not review_found:
            st.success("✅ 현재 자동 검수 기준에서 확인이 필요한 문장이 없습니다.")

    all_text_parts = []

    with tab_all:
        for item in st.session_state.v117_results:
            st.markdown("---")
            st.subheader(f"{item['index']}번 · {item['file_name']}")
            col_image, col_result = st.columns([0.85, 1.25], gap="large")

            with col_image:
                st.image(item["image"], caption="원본 이미지", use_container_width=True)
                st.caption(
                    f"실제 업로드 크기: {item['image'].width} × {item['image'].height}px"
                )
                if item["image"].width >= 800:
                    st.success("✅ 가로 800px 이상 · 해상도 경고 대상 아님")
                else:
                    st.warning("⚠ 가로 800px 미만 · 작은 글자 판독률이 떨어질 수 있음")

            with col_result:
                if item.get("error"):
                    st.error("번역 중 오류가 발생했습니다.")
                    st.code(item["error"])
                    continue

                segments = item.get("segments", [])
                if not segments:
                    st.warning("구조화된 번역 결과를 만들지 못했습니다.")
                    st.text_area(
                        "AI 원본 응답",
                        value=item.get("raw_response", ""),
                        height=300,
                        key=f"raw_{item['index']}",
                    )
                    continue

                edited_segments = []
                for seg_idx, segment in enumerate(segments, start=1):
                    status = "⚠ 확인" if segment.get("review_required") else "✅ 사용 가능"
                    length_status = (
                        f"⚠ 길이 {segment.get('length_ratio')}×"
                        if segment.get("too_long")
                        else f"✅ 길이 {segment.get('length_ratio')}×"
                    )
                    st.markdown(f"**{seg_idx}. {segment['type']} · {status} · {length_status}**")

                    korean_text = st.text_area(
                        "한국어 원문",
                        value=segment["korean"],
                        height=75,
                        key=f"ko_{item['index']}_{seg_idx}",
                    )
                    translated_text = st.text_area(
                        f"{display_language} 번역",
                        value=segment["translation"],
                        height=90,
                        key=f"tr_{item['index']}_{seg_idx}",
                    )

                    if segment.get("review_reason"):
                        st.caption(f"검수: {segment['review_reason']}")
                    if segment.get("dictionary_status"):
                        st.caption(f"📘 {segment['dictionary_status']}")

                    edited_segments.append(
                        {
                            **segment,
                            "order": seg_idx,
                            "korean": korean_text,
                            "translation": translated_text,
                        }
                    )

                item_text = segments_to_text(
                    item["file_name"],
                    selected_market,
                    display_language,
                    edited_segments,
                )
                all_text_parts.append(item_text)

                st.download_button(
                    label=f"📥 {item['index']}번 번역 TXT 다운로드",
                    data=item_text,
                    file_name=f"{item['index']:02d}_{selected_market}_translation.txt",
                    mime="text/plain",
                    key=f"download_{item['index']}",
                    use_container_width=True,
                )

    if all_text_parts:
        st.markdown("---")
        st.header("📚 전체 번역 결과")
        all_text = "\n\n".join(all_text_parts)
        st.download_button(
            label="📥 전체 번역 결과 TXT 다운로드",
            data=all_text,
            file_name=f"ALL_{selected_market}_translation.txt",
            mime="text/plain",
            use_container_width=True,
        )
