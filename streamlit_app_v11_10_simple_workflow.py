import base64
import io
import json
import re
from typing import Dict, List, Tuple

import streamlit as st
from openai import OpenAI
from PIL import Image

st.set_page_config(page_title="AI 상세페이지 번역기 v11.10", page_icon="🌐", layout="wide")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ------------------------------------------------------------
# Market presets
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
- 러시아 화장품 상세페이지에서 실제로 쓰는 짧고 자연스러운 카피를 사용하세요.
- 치료·치유·의학적 재생처럼 보이는 표현은 화장품 범위의 보습, 컨디셔닝, 외관 개선 표현으로 순화하세요.
- '완전히 안전', '자극 없음', '누구나 안심' 같은 절대적 안전 보장을 만들지 마세요.
- 저자극/민감성 관련 시험은 시험 범위 안에서만 표현하세요.
- 디자인용 문장은 불필요한 수식어를 줄여 짧게 번역하세요.
""",
    "미국": """
- 짧고 자연스러운 US cosmetic copy를 사용하세요.
- 질환 치료/예방, 신체 구조·기능 변화처럼 보이는 drug claim을 피하세요.
- cure, treat, heal, regenerate, 100% safe 같은 절대 표현을 만들지 마세요.
- helps, visibly, appearance of, feels 같은 화장품 범위 표현을 우선하세요.
""",
    "영국": """
- 자연스러운 British English 화장품 카피로 번역하세요.
- medicinal/therapeutic claim처럼 보일 수 있는 치료·치유·질환 예방 표현을 피하세요.
- 외관, 사용감, 보습 중심으로 보수적으로 표현하세요.
""",
    "글로벌 영어권": """
- 자연스러운 International English를 사용하세요.
- 치료·치유·질환 예방·절대 안전 보장 표현을 피하세요.
- helps, visibly, appearance of, feels 같은 화장품 범위 표현을 우선하세요.
""",
    "일본": """
- 일본 화장품 상세페이지에 자연스러운 짧은 카피로 현지화하세요.
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
st.title("🌐 AI 상세페이지 번역기 v11.10")
st.caption("원클릭 실무 모드: 국가 선택 → 이미지 업로드 → 번역 → 검수 필요한 문장만 확인 → 다운로드")
st.success("✅ v11.10 SIMPLE WORKFLOW · 고정 번역 사전/진단은 고급 설정에 숨김")

selected_market = st.selectbox("판매 국가 / 시장", list(MARKET_LANGUAGE.keys()), index=0)
target_language = MARKET_LANGUAGE[selected_market]
display_language = MARKET_DISPLAY_LANGUAGE[selected_market]

if "translation_dictionary" not in st.session_state:
    st.session_state.translation_dictionary = {}

with st.expander("⚙️ 고급 설정", expanded=False):
    safety_mode = st.checkbox("현지 화장품 광고 표현을 보수적으로 순화", value=True)
    short_mode = st.checkbox("디자인용 짧은 번역", value=True)
    show_diagnostics = st.checkbox("진단 탭 표시", value=False)
    st.caption("보통은 기본값 그대로 사용하면 됩니다.")

    st.markdown("---")
    st.markdown("#### 📘 고정 번역 사전")
    st.caption("필요할 때만 사용하세요. 한 번 확정한 표현을 계속 같은 번역으로 적용합니다.")
    default_dict_text = "\n".join(
        f"{k} = {v}" for k, v in st.session_state.translation_dictionary.items()
    )
    dictionary_text = st.text_area(
        "고정 번역",
        value=default_dict_text,
        height=120,
        placeholder="피부 저자극 테스트 완료 = Пройден тест на низкий потенциал раздражения кожи\n전성분 = 영문 INCI 유지",
        key="translation_dictionary_text",
    )
    parsed_dictionary = {}
    for line in dictionary_text.splitlines():
        if "=" not in line:
            continue
        source, target = line.split("=", 1)
        source, target = source.strip(), target.strip()
        if source and target:
            parsed_dictionary[source] = target
    st.session_state.translation_dictionary = parsed_dictionary

st.info(
    f"현재 프리셋: **{selected_market} / {display_language}** · "
    "광고 표현 순화 + 디자인용 짧은 번역"
)

uploaded_files = st.file_uploader(
    "상세페이지 JPG / PNG 이미지를 여러 장 업로드하세요",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def pil_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=94)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def normalize_key(text: str) -> str:
    return re.sub(r"[^가-힣A-Za-z0-9]", "", (text or "").lower())


def extract_numbers(text: str) -> List[str]:
    return re.findall(r"\d+(?:[.,]\d+)?%?", text or "")


def visual_units(text: str) -> float:
    units = 0.0
    for ch in text or "":
        if ch == "\n":
            continue
        if re.match(r"[가-힣一-龥ぁ-んァ-ン]", ch):
            units += 1.0
        elif ch.isspace():
            units += 0.35
        else:
            units += 0.55
    return max(1.0, units)


def length_check(korean: str, translation: str, seg_type: str) -> Dict:
    ratio = visual_units(translation) / visual_units(korean)
    if seg_type in {"headline", "label"}:
        limit = 1.35
    elif seg_type in {"footnote", "test_value"}:
        limit = 1.60
    else:
        limit = 1.50
    return {"ratio": round(ratio, 2), "limit": limit, "too_long": ratio > limit}


def deduplicate_segments(segments: List[Dict]) -> List[Dict]:
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


def split_tall_image(image: Image.Image, max_height: int = 3600, overlap: int = 220) -> List[Dict]:
    width, height = image.size
    if height <= max_height:
        return [{"index": 1, "y1": 0, "y2": height, "image": image}]

    chunks = []
    y1 = 0
    idx = 1
    while y1 < height:
        y2 = min(height, y1 + max_height)
        chunks.append({
            "index": idx,
            "y1": y1,
            "y2": y2,
            "image": image.crop((0, y1, width, y2)).convert("RGB"),
        })
        if y2 >= height:
            break
        y1 = max(0, y2 - overlap)
        idx += 1
    return chunks


def build_dictionary_prompt(dictionary: Dict[str, str]) -> str:
    if not dictionary:
        return "고정 번역 사전 없음"
    return "\n".join(f"- {k} => {v}" for k, v in dictionary.items())


def structured_schema() -> Dict:
    return {
        "type": "object",
        "properties": {
            "segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "korean": {"type": "string"},
                        "translation": {"type": "string"},
                        "type": {
                            "type": "string",
                            "enum": ["headline", "body", "label", "footnote", "test_value", "ingredients", "other"],
                        },
                        "review_required": {"type": "boolean"},
                        "review_reason": {"type": "string"},
                    },
                    "required": ["korean", "translation", "type", "review_required", "review_reason"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["segments"],
        "additionalProperties": False,
    }


def build_prompt(target_market: str, target_language: str, dictionary: Dict, safety_mode: bool, short_mode: bool) -> str:
    safety = (
        "원문보다 강한 효능·안전성 주장을 만들지 말고, 치료·완치·의학적 재생처럼 보이는 표현은 화장품 범위에서 보수적으로 순화하세요."
        if safety_mode else "원문 의미를 최대한 유지하세요."
    )
    concise = (
        "headline/label은 최대한 한 줄로, body는 1~2개의 짧은 문장으로 번역하고 불필요한 수식어를 줄이세요."
        if short_mode else "문장 길이를 임의로 과도하게 늘리지 마세요."
    )

    return f"""
이 이미지는 한국 화장품 상세페이지의 한 구간입니다.
이미지에 실제로 보이는 '상세페이지 한국어 카피'를 위에서 아래 순서로 모두 추출하고,
{target_market} 시장용 {target_language}로 자연스럽게 현지화하세요.

반드시 지킬 규칙:
1. 한국어 제목, 본문, 라벨, 작은 각주, 시험 수치 설명, 사용 전/사용 후 등을 확인하세요.
2. 제품 패키지에 인쇄된 글자, QR, 인증서/시험성적서 이미지 내부의 작은 인쇄물은 페이지 카피가 아니면 제외하세요.
3. 영어만 있는 브랜드명/제품명은 번역 대상으로 새로 만들지 마세요.
4. 실제 전체 전성분 목록은 번역하지 말고 한국어='전성분', 번역='영문 INCI 유지', type='ingredients'로 한 번만 반환하세요.
5. 숫자, %, ppm, ml, g, 날짜, 시험 수치는 그대로 유지하세요.
6. 여러 줄로 나뉜 하나의 문장은 하나로 합치세요.
7. 같은 한국어 문구를 중복 반환하지 마세요.
8. {safety}
9. {concise}
10. 시장별 규칙:\n{MARKET_RULES.get(target_market, '')}
11. 고정 번역 사전:\n{build_dictionary_prompt(dictionary)}
12. 판독이 불확실하거나 규제 순화 폭이 크거나 숫자가 복잡하면 review_required=true로 표시하세요.
13. 이미지 품질/해상도에 대한 안내문을 만들지 마세요.
14. 설명은 하지 말고 지정된 구조의 데이터만 반환하세요.
"""


def clean_segments(raw_segments: List[Dict], dictionary: Dict) -> List[Dict]:
    cleaned = []
    for seg in raw_segments or []:
        if not isinstance(seg, dict):
            continue
        korean = str(seg.get("korean", "")).strip()
        translation = str(seg.get("translation", "")).strip()
        seg_type = str(seg.get("type", "other")).strip() or "other"
        if not korean or not re.search(r"[가-힣]", korean):
            continue

        if korean in dictionary:
            translation = dictionary[korean]

        reasons = []
        ai_reason = str(seg.get("review_reason", "")).strip()
        if bool(seg.get("review_required", False)) and ai_reason:
            reasons.append(ai_reason)
        if extract_numbers(korean) != extract_numbers(translation) and seg_type != "ingredients":
            reasons.append("숫자/시험값 확인 필요")

        lc = length_check(korean, translation, seg_type)
        if lc["too_long"]:
            reasons.append(f"디자인 길이 초과 ({lc['ratio']}× / 권장 {lc['limit']}× 이하)")

        cleaned.append({
            "korean": korean,
            "translation": translation,
            "type": seg_type,
            "review_required": bool(reasons),
            "review_reason": " / ".join(dict.fromkeys(reasons)),
            "length_ratio": lc["ratio"],
            "length_limit": lc["limit"],
            "too_long": lc["too_long"],
        })
    return deduplicate_segments(cleaned)


def call_structured(image: Image.Image, prompt: str) -> Tuple[List[Dict], str]:
    """Primary path: Structured Outputs. Eliminates fragile JSON parsing."""
    response = client.responses.create(
        model="gpt-5",
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": pil_to_data_url(image), "detail": "high"},
            ],
        }],
        text={
            "format": {
                "type": "json_schema",
                "name": "detail_page_translation",
                "strict": True,
                "schema": structured_schema(),
            }
        },
        max_output_tokens=8000,
    )
    output = response.output_text or ""
    parsed = json.loads(output)
    return parsed.get("segments", []), output


def parse_plain_fallback(text: str) -> List[Dict]:
    """Fallback parser that does not depend on JSON at all."""
    result = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line.startswith("@@SEG@@"):
            continue
        payload = line[len("@@SEG@@"):].strip()
        parts = payload.split("|||", 5)
        if len(parts) < 5:
            continue
        if len(parts) == 5:
            seg_type, korean, translation, review, reason = parts
        else:
            seg_type, korean, translation, review, reason, _ = parts
        result.append({
            "type": seg_type.strip() or "other",
            "korean": korean.strip(),
            "translation": translation.strip(),
            "review_required": review.strip().lower() in {"1", "true", "yes", "y"},
            "review_reason": reason.strip(),
        })
    return result


def call_plain_fallback(image: Image.Image, prompt: str) -> Tuple[List[Dict], str]:
    fallback_prompt = prompt + """

Structured JSON 대신 이번에는 아래 한 줄 형식만 사용하세요.
문구 하나당 한 줄:
@@SEG@@ type|||한국어 원문|||번역문|||0 또는 1|||검수 사유

예:
@@SEG@@ headline|||피부 저자극 테스트 완료|||Пройден тест на низкий потенциал раздражения кожи|||0|||

다른 설명은 절대 쓰지 마세요.
"""
    response = client.responses.create(
        model="gpt-5",
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": fallback_prompt},
                {"type": "input_image", "image_url": pil_to_data_url(image), "detail": "high"},
            ],
        }],
        max_output_tokens=8000,
    )
    output = response.output_text or ""
    return parse_plain_fallback(output), output


def extract_korean_only(image: Image.Image) -> Tuple[List[Dict], str]:
    """Last-resort extraction. It only asks for Korean copy, then translation is done separately."""
    prompt = """
이 한국 화장품 상세페이지 이미지에서 실제 페이지 위에 배치된 한국어 카피를 위에서 아래 순서로 추출하세요.
제품 패키지 내부 글자, QR코드, 시험성적서/인증서 이미지 내부 작은 글자는 제외하세요.
작은 각주, 사용 전/사용 후, 숫자가 포함된 시험 설명도 포함하세요.
문장 하나당 반드시 한 줄로만 출력하세요.
형식: @@KO@@ 한국어 문구
한국어가 보이는데 누락하지 않도록 전체 이미지를 확인하세요.
다른 설명은 쓰지 마세요.
"""
    response = client.responses.create(
        model="gpt-5",
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": pil_to_data_url(image), "detail": "high"},
            ],
        }],
        max_output_tokens=5000,
    )
    output = response.output_text or ""
    items = []
    for line in output.splitlines():
        line = line.strip()
        if not line.startswith("@@KO@@"):
            continue
        korean = line[len("@@KO@@"):].strip()
        if re.search(r"[가-힣]", korean):
            items.append({"korean": korean, "type": "other"})
    return items, output


def translate_extracted_text(items: List[Dict], prompt: str) -> Tuple[List[Dict], str]:
    if not items:
        return [], ""
    source_lines = "\n".join(f"{i+1}. {item['korean']}" for i, item in enumerate(items))
    text_prompt = prompt + f"""

이미지 판독은 이미 끝났습니다. 아래 한국어 목록만 번역하세요.
각 번호를 빠짐없이 유지하고 다음 형식으로 한 줄씩 반환하세요:
@@TR@@ 번호|||type|||한국어 원문|||번역문|||0 또는 1|||검수 사유

한국어 목록:
{source_lines}
"""
    response = client.responses.create(model="gpt-5", input=text_prompt, max_output_tokens=8000)
    output = response.output_text or ""
    result = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line.startswith("@@TR@@"):
            continue
        payload = line[len("@@TR@@"):].strip()
        parts = payload.split("|||", 5)
        if len(parts) < 5:
            continue
        if len(parts) == 5:
            _, seg_type, korean, translation, review = parts
            reason = ""
        else:
            _, seg_type, korean, translation, review, reason = parts
        result.append({
            "type": seg_type.strip() or "other",
            "korean": korean.strip(),
            "translation": translation.strip(),
            "review_required": review.strip().lower() in {"1", "true", "yes", "y"},
            "review_reason": reason.strip(),
        })
    return result, output


def translate_chunk(image: Image.Image, target_market: str, target_language: str, dictionary: Dict, safety_mode: bool, short_mode: bool) -> Tuple[List[Dict], Dict]:
    prompt = build_prompt(target_market, target_language, dictionary, safety_mode, short_mode)
    report = {"method": "", "error": "", "raw_preview": ""}

    # 1) Structured Outputs
    try:
        segments, raw = call_structured(image, prompt)
        cleaned = clean_segments(segments, dictionary)
        if cleaned:
            report.update({"method": "structured", "raw_preview": raw[:700]})
            return cleaned, report
        report["error"] = "structured returned 0 segments"
    except Exception as exc:
        report["error"] = f"structured error: {exc}"

    # 2) Plain delimiter fallback
    try:
        segments, raw = call_plain_fallback(image, prompt)
        cleaned = clean_segments(segments, dictionary)
        if cleaned:
            report.update({"method": "plain_fallback", "raw_preview": raw[:700]})
            return cleaned, report
        report["error"] += " | plain fallback returned 0"
    except Exception as exc:
        report["error"] += f" | plain fallback error: {exc}"

    # 3) Extraction-only fallback + text-only translation
    try:
        ko_items, raw_extract = extract_korean_only(image)
        translated, raw_translate = translate_extracted_text(ko_items, prompt)
        cleaned = clean_segments(translated, dictionary)
        if cleaned:
            report.update({
                "method": "extract_then_translate",
                "raw_preview": (raw_extract + "\n---\n" + raw_translate)[:700],
            })
            return cleaned, report
        report["error"] += " | extract-translate returned 0"
    except Exception as exc:
        report["error"] += f" | extract-translate error: {exc}"

    report["method"] = "failed"
    return [], report


def translate_image_auto_split(image: Image.Image, target_market: str, target_language: str, dictionary: Dict, safety_mode: bool, short_mode: bool) -> Tuple[List[Dict], List[Dict]]:
    chunks = split_tall_image(image, max_height=3600, overlap=220)
    all_segments = []
    reports = []

    for chunk in chunks:
        segments, report = translate_chunk(
            chunk["image"], target_market, target_language, dictionary, safety_mode, short_mode
        )
        report.update({"chunk": chunk["index"], "y1": chunk["y1"], "y2": chunk["y2"], "segments": len(segments)})
        reports.append(report)
        for seg in segments:
            item = dict(seg)
            item["chunk"] = chunk["index"]
            all_segments.append(item)

    all_segments = deduplicate_segments(all_segments)
    return all_segments, reports


def segments_to_text(file_name: str, segments: List[Dict]) -> str:
    parts = [f"===== {file_name} =====", f"시장: {selected_market} / {display_language}", ""]
    for seg in segments:
        parts.extend([
            f"[{seg['order']}] {seg['type']}",
            "[한국어]",
            seg["korean"],
            "",
            f"[{display_language}]",
            seg["translation"],
            "",
            f"검수: {'필요' if seg.get('review_required') else '바로 사용'} {seg.get('review_reason', '')}",
            "",
        ])
    return "\n".join(parts)

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
if "v119_results" not in st.session_state:
    st.session_state.v119_results = []

if uploaded_files and st.button("🚀 전체 이미지 번역 시작", type="primary"):
    results = []
    progress = st.progress(0)

    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        uploaded_file.seek(0)
        image = Image.open(uploaded_file).convert("RGB")

        with st.spinner(f"{idx}/{len(uploaded_files)} · {uploaded_file.name} 번역 중..."):
            try:
                segments, reports = translate_image_auto_split(
                    image,
                    selected_market,
                    target_language,
                    st.session_state.translation_dictionary,
                    safety_mode,
                    short_mode,
                )
                results.append({
                    "index": idx,
                    "file_name": uploaded_file.name,
                    "image": image,
                    "segments": segments,
                    "reports": reports,
                })
            except Exception as exc:
                results.append({
                    "index": idx,
                    "file_name": uploaded_file.name,
                    "image": image,
                    "segments": [],
                    "reports": [{"method": "fatal", "error": str(exc), "segments": 0}],
                })
        progress.progress(idx / len(uploaded_files))

    st.session_state.v119_results = results

    zero = [r for r in results if not r["segments"]]
    if zero:
        st.warning(f"⚠️ {len(zero)}개 이미지에서 문장을 추출하지 못했습니다. 아래 진단 정보를 확인해 주세요.")
    else:
        st.success("✅ 모든 이미지 번역 완료")

if st.session_state.v119_results:
    st.markdown("---")
    total = sum(len(r["segments"]) for r in st.session_state.v119_results)
    review_count = sum(1 for r in st.session_state.v119_results for s in r["segments"] if s.get("review_required"))
    too_long = sum(1 for r in st.session_state.v119_results for s in r["segments"] if s.get("too_long"))

    c1, c2, c3 = st.columns(3)
    c1.metric("전체 문장", total)
    c2.metric("⚠ 검수 필요", review_count)
    c3.metric("✂ 길이 초과", too_long)

    if show_diagnostics:
        tab_review, tab_all, tab_diag = st.tabs(
            ["⚠ 검수 필요한 문장만", "📑 전체 번역", "🛠 진단"]
        )
    else:
        tab_review, tab_all = st.tabs(["⚠ 검수 필요한 문장만", "📑 전체 번역"])
        tab_diag = None

    with tab_review:
        reviews = [(r, s) for r in st.session_state.v119_results for s in r["segments"] if s.get("review_required")]
        if not reviews:
            st.success("현재 자동 검수 기준에서 확인이 필요한 문장이 없습니다.")
        for r, seg in reviews:
            st.markdown(f"**{r['file_name']} · {seg['order']}번**")
            st.write(seg["korean"])
            st.write(seg["translation"])
            st.caption(seg.get("review_reason", ""))
            st.markdown("---")

    all_text_parts = []
    with tab_all:
        for r in st.session_state.v119_results:
            st.subheader(f"{r['index']}번 · {r['file_name']}")
            if not r["segments"]:
                st.error("문장 추출 실패 — 진단 탭을 확인해 주세요.")
                continue

            left, right = st.columns([0.75, 1.25], gap="large")
            with left:
                st.image(r["image"], use_container_width=True)
            with right:
                edited = []
                for seg in r["segments"]:
                    st.markdown(f"**{seg['order']}. {seg['type']}**")
                    ko = st.text_area("한국어 원문", value=seg["korean"], height=75, key=f"ko_{r['index']}_{seg['order']}")
                    tr = st.text_area(display_language, value=seg["translation"], height=90, key=f"tr_{r['index']}_{seg['order']}")
                    if seg.get("review_required"):
                        st.warning(seg.get("review_reason", "검수 필요"))
                    edited.append({**seg, "korean": ko, "translation": tr})

                txt = segments_to_text(r["file_name"], edited)
                all_text_parts.append(txt)
                st.download_button(
                    f"📥 {r['index']}번 번역 TXT 다운로드",
                    data=txt,
                    file_name=f"{r['index']:02d}_{selected_market}_translation.txt",
                    mime="text/plain",
                    key=f"download_{r['index']}",
                    use_container_width=True,
                )

        if all_text_parts:
            st.markdown("---")
            all_txt = "\n\n".join(all_text_parts)
            st.download_button(
                "📥 전체 번역 결과 TXT 다운로드",
                data=all_txt,
                file_name=f"ALL_{selected_market}_translation.txt",
                mime="text/plain",
                use_container_width=True,
            )

    if show_diagnostics and tab_diag is not None:
        with tab_diag:
            st.caption("문장 0개가 나오면 어떤 fallback까지 실행됐는지 확인할 수 있습니다.")
            for r in st.session_state.v119_results:
                st.markdown(f"### {r['index']}번 · {r['file_name']}")
                for report in r["reports"]:
                    method = report.get("method", "")
                    seg_count = report.get("segments", 0)
                    error = report.get("error", "")
                    st.write(
                        f"조각 {report.get('chunk', '-')} · 방법: **{method}** · 문장: **{seg_count}개**"
                    )
                    if error:
                        st.code(error)
                    if report.get("raw_preview"):
                        with st.expander("AI 응답 일부 보기"):
                            st.code(report["raw_preview"])
