import base64
import csv
import io
import json
import re
from typing import Dict, List, Tuple

import streamlit as st
from openai import OpenAI
from PIL import Image

st.set_page_config(
    page_title="AI 상세페이지 번역기 v12",
    page_icon="🌐",
    layout="wide",
)

# ------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------
API_KEY = st.secrets.get("OPENAI_API_KEY", "")
if not API_KEY:
    st.error("OPENAI_API_KEY가 없습니다. Streamlit Secrets에 기존 API 키를 넣어주세요.")
    st.stop()

client = OpenAI(api_key=API_KEY)

MODEL_ECONOMY = "gpt-5-mini"
# 2026-08 기준 GPT-5 mini 공식 단가(대략 비용 표시용)
INPUT_USD_PER_1M = 0.25
OUTPUT_USD_PER_1M = 2.00

MARKET_LANGUAGE = {
    "러시아": ("Russian", "러시아어"),
    "미국": ("American English", "영어"),
    "영국": ("British English", "영어"),
    "글로벌 영어권": ("International English", "영어"),
    "일본": ("Japanese", "일본어"),
    "중국 본토": ("Simplified Chinese", "중국어(간체)"),
    "대만": ("Traditional Chinese", "중국어(번체)"),
    "베트남": ("Vietnamese", "베트남어"),
    "프랑스": ("French", "프랑스어"),
    "스페인": ("Spanish (Spain)", "스페인어"),
    "멕시코": ("Spanish (Mexico)", "스페인어"),
    "라틴아메리카": ("Neutral Latin American Spanish", "스페인어"),
}

MARKET_RULES = {
    "러시아": "러시아 화장품 상세페이지에서 자연스러운 짧은 표현을 사용한다. 절대적 안전 보장, 치료/의약품 표현은 피한다.",
    "미국": "cosmetic claim 범위를 유지하고 drug/treatment/cure/medical regeneration처럼 들리는 표현을 피한다.",
    "영국": "UK cosmetic advertising에 자연스러운 표현을 사용하고 의약품성·절대적 효능 표현을 피한다.",
    "글로벌 영어권": "국가 특유의 속어를 피하고 국제적으로 자연스러운 짧은 화장품 카피를 사용한다.",
    "일본": "化粧品 표현 범위를 유지하며 肌を整える・うるおい・ハリ感 등 자연스러운 화장품 표현을 우선한다.",
    "중국 본토": "절대적/최상급 표현(最, 第一, 100%, 零刺激 등)을 새로 만들지 않고 화장품 범위로 순화한다.",
    "대만": "번체 중국어를 사용하고 의약품성·절대적 안전 보장 표현을 피한다.",
    "베트남": "베트남 화장품 상세페이지에 자연스러운 짧은 표현을 사용하며 치료성·절대 안전 표현을 피한다.",
    "프랑스": "프랑스/EU 화장품 범위의 자연스러운 표현을 사용하고 의약품성·절대적 효능/안전 보장을 피한다.",
    "스페인": "스페인 화장품 상세페이지에 자연스러운 표현을 사용하고 치료성·절대적 효능 표현을 피한다.",
    "멕시코": "멕시코에서 자연스러운 화장품 스페인어를 사용하고 치료성·절대적 안전 표현을 피한다.",
    "라틴아메리카": "중립적인 라틴아메리카 스페인어를 사용하고 치료성·절대적 효능 표현을 피한다.",
}

TYPE_LIMIT = {
    "headline": 1.30,
    "label": 1.30,
    "footnote": 1.55,
    "test_value": 1.55,
    "body": 1.50,
    "ingredients": 1.00,
    "other": 1.50,
}

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.title("🌐 AI 상세페이지 번역기 v12")
st.success("✅ STABLE LOW-COST · NO-OCR · GPT-5 mini · 긴 이미지 자동분할")
st.caption("국가 선택 → 이미지 업로드 → 번역 → 검수 필요한 문장만 확인 → 다운로드")

selected_market = st.selectbox("판매 국가 / 시장", list(MARKET_LANGUAGE.keys()), index=0)
target_language, display_language = MARKET_LANGUAGE[selected_market]

with st.expander("⚙️ 고급 설정", expanded=False):
    safety_mode = st.checkbox("광고 표현 보수적으로 순화", value=True)
    short_mode = st.checkbox("디자인용 짧은 번역", value=True)
    retry_failed = st.checkbox("실패한 조각만 1회 자동 재시도", value=True)
    chunk_height = st.select_slider(
        "긴 이미지 자동 분할 높이",
        options=[2200, 2600, 3000, 3400, 3800],
        value=3000,
        help="세로가 긴 상세페이지를 이 높이 기준으로 자동 분할합니다.",
    )
    dictionary_text = st.text_area(
        "고정 번역 사전 (선택)",
        value="피부 저자극 테스트 완료 = Пройден тест на низкий потенциал раздражения кожи\n전성분 = 영문 INCI 유지" if selected_market == "러시아" else "전성분 = 영문 INCI 유지",
        height=100,
        help="한 줄에 '한국어 = 번역문' 형식으로 입력하세요.",
    )

uploaded_files = st.file_uploader(
    "상세페이지 JPG / PNG 이미지를 여러 장 올려주세요",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

# ------------------------------------------------------------
# 유틸
# ------------------------------------------------------------
def parse_dictionary(text: str) -> Dict[str, str]:
    result = {}
    for raw in (text or "").splitlines():
        if "=" not in raw:
            continue
        left, right = raw.split("=", 1)
        left = left.strip()
        right = right.strip()
        if left and right:
            result[left] = right
    return result


def pil_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=92, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


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


def split_tall_image(image: Image.Image, max_height: int, overlap: int = 180) -> List[Dict]:
    width, height = image.size
    if height <= max_height:
        return [{"index": 1, "y1": 0, "y2": height, "image": image.convert("RGB")}]

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


def build_schema() -> Dict:
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


def build_prompt(market: str, language: str, dictionary: Dict[str, str], safety: bool, short: bool) -> str:
    dict_text = "\n".join(f"- {k} => {v}" for k, v in dictionary.items()) or "없음"
    safety_text = (
        "원문보다 강한 효능/안전 주장을 만들지 않는다. 치료·완치·의학적 재생·절대적 안전 보장처럼 보이는 표현은 화장품 범위에서만 최소한으로 순화한다."
        if safety else
        "원문 의미를 최대한 그대로 유지한다."
    )
    short_text = (
        "디자인에 바로 넣을 수 있게 짧게 번역한다. headline/label은 한 줄 우선, body는 원문과 비슷한 시각적 길이를 목표로 하며 불필요한 수식어를 추가하지 않는다."
        if short else
        "원문보다 불필요하게 길어지지 않게 번역한다."
    )

    return f"""
이 이미지는 한국 화장품 상세페이지의 한 구간이다.
이미지에 실제로 보이는 '페이지 위 한국어 카피'만 위에서 아래 순서로 모두 추출하고,
{market} 시장용 {language}로 자연스럽게 현지화한다.

필수 규칙:
1. 제목, 본문, 라벨, 작은 각주, 테스트 문구, 숫자 설명, 사용 전/사용 후 등 실제 페이지 카피를 확인한다.
2. 제품 패키지에 인쇄된 글자, QR코드, 인증서/시험성적서/사진/삽입 그래픽 내부의 작은 인쇄물은 페이지 카피가 아니면 제외한다.
3. 영어만 있는 브랜드명/제품명은 새 번역 대상으로 만들지 않는다.
4. 여러 줄로 나뉜 하나의 문장은 하나로 합친다.
5. 같은 한국어 문구를 중복 반환하지 않는다.
6. 숫자, %, ppm, ml, g, 날짜, 시험 수치는 원문 그대로 보존한다.
7. 실제 전체 전성분 목록은 번역하지 않고 korean='전성분', translation='영문 INCI 유지', type='ingredients'로 한 번만 반환한다.
8. {safety_text}
9. {short_text}
10. 시장 규칙: {MARKET_RULES.get(market, '')}
11. 고정 사전:\n{dict_text}
12. 판독이 애매하거나 규제 순화 폭이 크거나 숫자 확인이 필요하면 review_required=true로 한다.
13. 이미지 품질/해상도에 대한 안내문을 결과로 만들지 않는다.
14. 지정된 JSON 구조 외 설명을 쓰지 않는다.
"""


def response_usage(response) -> Tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    return int(getattr(usage, "input_tokens", 0) or 0), int(getattr(usage, "output_tokens", 0) or 0)


def call_structured(image: Image.Image, prompt: str) -> Tuple[List[Dict], Dict]:
    response = client.responses.create(
        model=MODEL_ECONOMY,
        store=False,
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
                "schema": build_schema(),
            }
        },
        max_output_tokens=5000,
    )
    raw = response.output_text or ""
    parsed = json.loads(raw)
    in_tok, out_tok = response_usage(response)
    return parsed.get("segments", []), {
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "raw": raw[:800],
    }


def call_retry_plain(image: Image.Image, prompt: str) -> Tuple[List[Dict], Dict]:
    retry_prompt = prompt + """

첫 시도 형식 처리에 실패했다. 이번에는 반드시 다음 형식으로만 출력한다.
문구 하나당 한 줄:
@@SEG@@ type|||한국어 원문|||번역문|||0 또는 1|||검수 사유
예: @@SEG@@ headline|||피부 저자극 테스트 완료|||번역문|||0|||
다른 설명은 쓰지 않는다.
"""
    response = client.responses.create(
        model=MODEL_ECONOMY,
        store=False,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": retry_prompt},
                {"type": "input_image", "image_url": pil_to_data_url(image), "detail": "high"},
            ],
        }],
        max_output_tokens=5000,
    )
    raw = response.output_text or ""
    segments = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("@@SEG@@"):
            continue
        parts = line[len("@@SEG@@"):].strip().split("|||", 4)
        if len(parts) != 5:
            continue
        seg_type, korean, translation, review, reason = parts
        segments.append({
            "type": seg_type.strip() or "other",
            "korean": korean.strip(),
            "translation": translation.strip(),
            "review_required": review.strip().lower() in {"1", "true", "yes", "y"},
            "review_reason": reason.strip(),
        })
    in_tok, out_tok = response_usage(response)
    return segments, {
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "raw": raw[:800],
    }


def clean_segments(raw_segments: List[Dict], dictionary: Dict[str, str]) -> List[Dict]:
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

        if seg_type != "ingredients" and extract_numbers(korean) != extract_numbers(translation):
            reasons.append("숫자/시험값 확인 필요")

        ratio = round(visual_units(translation) / visual_units(korean), 2)
        limit = TYPE_LIMIT.get(seg_type, 1.50)
        if ratio > limit:
            reasons.append(f"디자인 길이 초과 {ratio}× (권장 ≤ {limit}×)")

        cleaned.append({
            "korean": korean,
            "translation": translation,
            "type": seg_type,
            "review_required": bool(reasons),
            "review_reason": " / ".join(dict.fromkeys(reasons)),
            "length_ratio": ratio,
            "too_long": ratio > limit,
        })
    return cleaned


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


def translate_chunk(image: Image.Image, prompt: str, dictionary: Dict[str, str], retry: bool) -> Tuple[List[Dict], Dict]:
    total_in = 0
    total_out = 0
    try:
        raw_segments, usage = call_structured(image, prompt)
        total_in += usage["input_tokens"]
        total_out += usage["output_tokens"]
        cleaned = clean_segments(raw_segments, dictionary)
        if cleaned:
            return cleaned, {
                "status": "ok",
                "method": "structured",
                "error": "",
                "input_tokens": total_in,
                "output_tokens": total_out,
            }
        first_error = "구조화 응답에서 문장 0개"
    except Exception as exc:
        first_error = f"구조화 응답 오류: {exc}"

    if not retry:
        return [], {
            "status": "failed",
            "method": "structured",
            "error": first_error,
            "input_tokens": total_in,
            "output_tokens": total_out,
        }

    try:
        raw_segments, usage = call_retry_plain(image, prompt)
        total_in += usage["input_tokens"]
        total_out += usage["output_tokens"]
        cleaned = clean_segments(raw_segments, dictionary)
        if cleaned:
            return cleaned, {
                "status": "ok_retry",
                "method": "plain_retry",
                "error": first_error,
                "input_tokens": total_in,
                "output_tokens": total_out,
            }
        retry_error = "재시도에서도 문장 0개"
    except Exception as exc:
        retry_error = f"재시도 오류: {exc}"

    return [], {
        "status": "failed",
        "method": "failed",
        "error": f"{first_error} | {retry_error}",
        "input_tokens": total_in,
        "output_tokens": total_out,
    }


def translate_image(image: Image.Image, prompt: str, dictionary: Dict[str, str], max_height: int, retry: bool) -> Tuple[List[Dict], List[Dict], int, int]:
    chunks = split_tall_image(image, max_height=max_height, overlap=180)
    all_segments = []
    reports = []
    total_in = 0
    total_out = 0

    for chunk in chunks:
        segments, report = translate_chunk(chunk["image"], prompt, dictionary, retry)
        report.update({
            "chunk": chunk["index"],
            "y1": chunk["y1"],
            "y2": chunk["y2"],
            "segments": len(segments),
        })
        reports.append(report)
        total_in += report.get("input_tokens", 0)
        total_out += report.get("output_tokens", 0)
        all_segments.extend(segments)

    return deduplicate_segments(all_segments), reports, total_in, total_out


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1_000_000 * INPUT_USD_PER_1M) + (output_tokens / 1_000_000 * OUTPUT_USD_PER_1M)


def segments_to_txt(file_name: str, segments: List[Dict]) -> str:
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


def segments_to_csv(file_name: str, segments: List[Dict]) -> bytes:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["image", "order", "type", "korean", display_language, "review", "reason"])
    for seg in segments:
        writer.writerow([
            file_name,
            seg["order"],
            seg["type"],
            seg["korean"],
            seg["translation"],
            "검수 필요" if seg.get("review_required") else "바로 사용",
            seg.get("review_reason", ""),
        ])
    return output.getvalue().encode("utf-8-sig")

# ------------------------------------------------------------
# 실행 전 예상 호출 수
# ------------------------------------------------------------
dictionary = parse_dictionary(dictionary_text)

if uploaded_files:
    preview_rows = []
    estimated_calls = 0
    for file in uploaded_files:
        file.seek(0)
        img = Image.open(file)
        chunk_count = len(split_tall_image(img, max_height=chunk_height, overlap=180))
        estimated_calls += chunk_count
        preview_rows.append(f"{file.name}: {img.width}×{img.height}px → {chunk_count}조각")
        file.seek(0)

    st.info(
        " · ".join(preview_rows) +
        f"\n\n기본 API 호출 예상: **{estimated_calls}회** " +
        ("(실패 조각만 최대 1회 추가 재시도)" if retry_failed else "(자동 재시도 없음)")
    )

    if st.button("🚀 번역 시작", type="primary", use_container_width=True):
        results = []
        progress = st.progress(0)
        total_files = len(uploaded_files)

        prompt = build_prompt(selected_market, target_language, dictionary, safety_mode, short_mode)

        for idx, file in enumerate(uploaded_files, start=1):
            file.seek(0)
            image = Image.open(file).convert("RGB")
            with st.spinner(f"{idx}/{total_files} · {file.name} 번역 중..."):
                try:
                    segments, reports, in_tok, out_tok = translate_image(
                        image, prompt, dictionary, chunk_height, retry_failed
                    )
                    results.append({
                        "index": idx,
                        "file_name": file.name,
                        "image": image,
                        "segments": segments,
                        "reports": reports,
                        "input_tokens": in_tok,
                        "output_tokens": out_tok,
                        "cost": estimate_cost(in_tok, out_tok),
                    })
                except Exception as exc:
                    results.append({
                        "index": idx,
                        "file_name": file.name,
                        "image": image,
                        "segments": [],
                        "reports": [{"status": "fatal", "error": str(exc), "segments": 0}],
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cost": 0.0,
                    })
            progress.progress(idx / total_files)

        st.session_state["v12_results"] = results

# ------------------------------------------------------------
# 결과
# ------------------------------------------------------------
if st.session_state.get("v12_results"):
    results = st.session_state["v12_results"]
    st.markdown("---")

    total_segments = sum(len(r["segments"]) for r in results)
    review_count = sum(1 for r in results for s in r["segments"] if s.get("review_required"))
    failed_chunks = sum(1 for r in results for rep in r["reports"] if rep.get("status") == "failed")
    total_in = sum(r["input_tokens"] for r in results)
    total_out = sum(r["output_tokens"] for r in results)
    total_cost = sum(r["cost"] for r in results)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("전체 문장", total_segments)
    c2.metric("⚠ 검수 필요", review_count)
    c3.metric("실패 조각", failed_chunks)
    c4.metric("이번 실행 예상 API 비용", f"${total_cost:.4f}")

    st.caption(f"이번 실행 토큰: 입력 {total_in:,} / 출력 {total_out:,} · 모델 {MODEL_ECONOMY}")

    if total_segments == 0:
        st.error("문장을 하나도 추출하지 못했습니다. 아래 진단에서 오류 내용을 확인해 주세요.")
    elif failed_chunks:
        st.warning("일부 조각은 실패했지만 나머지 번역 결과는 사용할 수 있습니다. 실패 조각만 다시 시도하면 됩니다.")
    else:
        st.success("✅ 번역 완료")

    tab_review, tab_all, tab_diag = st.tabs(["⚠ 검수 필요한 문장", "📄 전체 번역", "🔧 진단"])

    with tab_review:
        review_items = [(r, s) for r in results for s in r["segments"] if s.get("review_required")]
        if not review_items:
            st.success("현재 자동 검수 기준에서 확인이 필요한 문장이 없습니다.")
        for r, seg in review_items:
            st.markdown(f"**{r['file_name']} · {seg['order']}번**")
            st.write(seg["korean"])
            st.text_area(
                display_language,
                value=seg["translation"],
                key=f"review_{r['index']}_{seg['order']}",
                height=80,
            )
            st.caption(seg.get("review_reason", ""))
            st.markdown("---")

    with tab_all:
        all_txt_parts = []
        all_csv_rows = []

        for r in results:
            st.subheader(f"{r['index']}번 · {r['file_name']}")
            if not r["segments"]:
                st.error("이 이미지에서는 번역 문장을 만들지 못했습니다.")
                continue

            edited_segments = []
            for seg in r["segments"]:
                left, right = st.columns([1, 1])
                with left:
                    ko = st.text_area(
                        "한국어 원문",
                        value=seg["korean"],
                        key=f"ko_{r['index']}_{seg['order']}",
                        height=75,
                    )
                with right:
                    tr = st.text_area(
                        display_language,
                        value=seg["translation"],
                        key=f"tr_{r['index']}_{seg['order']}",
                        height=75,
                    )
                edited = dict(seg)
                edited["korean"] = ko
                edited["translation"] = tr
                edited_segments.append(edited)

            txt = segments_to_txt(r["file_name"], edited_segments)
            csv_bytes = segments_to_csv(r["file_name"], edited_segments)
            all_txt_parts.append(txt)

            d1, d2 = st.columns(2)
            d1.download_button(
                "📥 TXT 다운로드",
                data=txt,
                file_name=f"{r['index']:02d}_{display_language}_translation.txt",
                mime="text/plain",
                key=f"txt_{r['index']}",
                use_container_width=True,
            )
            d2.download_button(
                "📥 CSV 다운로드",
                data=csv_bytes,
                file_name=f"{r['index']:02d}_{display_language}_translation.csv",
                mime="text/csv",
                key=f"csv_{r['index']}",
                use_container_width=True,
            )
            st.markdown("---")

        if all_txt_parts:
            st.download_button(
                "📦 전체 번역 TXT 다운로드",
                data="\n\n".join(all_txt_parts),
                file_name=f"ALL_{display_language}_translation.txt",
                mime="text/plain",
                use_container_width=True,
            )

    with tab_diag:
        for r in results:
            st.markdown(f"**{r['file_name']}**")
            for rep in r["reports"]:
                status = rep.get("status", "unknown")
                chunk = rep.get("chunk", "-")
                segs = rep.get("segments", 0)
                method = rep.get("method", "-")
                st.write(f"조각 {chunk} · {status} · {method} · 문장 {segs}개")
                if rep.get("error"):
                    st.code(rep["error"])
else:
    st.info("이미지를 올리면 예상 분할 수와 API 호출 수를 먼저 보여줍니다.")
