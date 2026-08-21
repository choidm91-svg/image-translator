import base64
import csv
import io
import json
import re
from typing import Dict, List, Tuple

import streamlit as st
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont, ImageStat, ImageFilter

st.set_page_config(
    page_title="AI 상세페이지 번역기 v12.2",
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
st.title("🌐 AI 상세페이지 번역기 v12.2")
st.success("✅ v12.2 LOW-COST · NO-OCR · 번역 + 줄바꿈 + 원본 텍스트 자동 교체")
st.caption("국가 선택 → 이미지 업로드 → 번역 → 줄바꿈 → 번역 이미지 생성 → PNG/JPG 다운로드")

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
    image_replace_mode = st.selectbox(
        "번역 이미지 자동 교체 방식",
        ["안전형 (추천)", "적극형"],
        index=0,
        help="안전형은 사진/복잡한 배경 위 문구를 억지로 지우지 않습니다. 적극형은 단순 추정 배경으로도 교체를 시도합니다.",
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


def make_preview_image(image: Image.Image, max_width: int = 420, max_height: int = 900) -> Image.Image:
    preview = image.convert("RGB").copy()
    preview.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
    return preview


def normalize_linebreak_compare(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def local_design_linebreak(text: str, seg_type: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    units = visual_units(text)
    if seg_type in {"headline", "label"}:
        target_lines = 1 if units <= 24 else 2
    elif seg_type in {"footnote", "test_value"}:
        target_lines = 1 if units <= 32 else 2
    else:
        target_lines = 1 if units <= 34 else (2 if units <= 58 else 3)

    if target_lines == 1:
        return text

    words = text.split()
    if len(words) >= target_lines:
        lines = []
        start = 0
        remaining_units = visual_units(text)
        for line_idx in range(target_lines - 1):
            remaining_lines = target_lines - line_idx
            target = remaining_units / remaining_lines
            best_end = start + 1
            best_score = float("inf")
            max_end = len(words) - (remaining_lines - 1)
            for end in range(start + 1, max_end + 1):
                candidate = " ".join(words[start:end])
                score = abs(visual_units(candidate) - target)
                if candidate.endswith((",", ":", ";", "—", "–")):
                    score -= 0.8
                if score < best_score:
                    best_score = score
                    best_end = end
            line = " ".join(words[start:best_end]).strip()
            lines.append(line)
            remaining_units -= visual_units(line)
            start = best_end
        lines.append(" ".join(words[start:]).strip())
        return "\n".join(line for line in lines if line)

    chars = list(text)
    approx = max(1, len(chars) // target_lines)
    lines = []
    start = 0
    for line_idx in range(target_lines - 1):
        ideal = min(len(chars) - 1, start + approx)
        search_start = max(start + 1, ideal - 5)
        search_end = min(len(chars) - (target_lines - line_idx - 1), ideal + 6)
        break_at = ideal
        for pos in range(search_start, search_end):
            if chars[pos - 1] in "、，。！？；：":
                break_at = pos
                break
        lines.append("".join(chars[start:break_at]).strip())
        start = break_at
    lines.append("".join(chars[start:]).strip())
    return "\n".join(line for line in lines if line)



def valid_bbox_norm(box) -> bool:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    try:
        x1, y1, x2, y2 = [int(v) for v in box]
    except Exception:
        return False
    return 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000


def clamp_box(box, width: int, height: int):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def parse_hex_color(value: str, fallback=(32, 32, 32)):
    value = (value or '').strip().lstrip('#')
    if len(value) == 6:
        try:
            return tuple(int(value[i:i+2], 16) for i in (0, 2, 4))
        except Exception:
            pass
    return fallback


def luminance(rgb):
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def side_sample_stats(image: Image.Image, box, margin: int = 10):
    """텍스트 박스 바깥의 얇은 띠만 샘플링하여 배경색/복잡도를 추정합니다."""
    image = image.convert('RGB')
    w, h = image.size
    x1, y1, x2, y2 = clamp_box(box, w, h)
    m = max(3, min(margin, max(3, min(x2-x1, y2-y1)//3)))

    regions = []
    if y1 > 0:
        regions.append(('top', image.crop((x1, max(0, y1-m), x2, y1))))
    if y2 < h:
        regions.append(('bottom', image.crop((x1, y2, x2, min(h, y2+m)))))
    if x1 > 0:
        regions.append(('left', image.crop((max(0, x1-m), y1, x1, y2))))
    if x2 < w:
        regions.append(('right', image.crop((x2, y1, min(w, x2+m), y2))))

    medians = {}
    side_noise = []
    for name, region in regions:
        if region.width <= 0 or region.height <= 0:
            continue
        stat = ImageStat.Stat(region)
        med = tuple(int(v) for v in stat.median[:3])
        std = sum(stat.stddev[:3]) / 3.0
        medians[name] = med
        side_noise.append(std)

    if medians:
        vals = list(medians.values())
        overall = tuple(int(sum(v[i] for v in vals) / len(vals)) for i in range(3))
    else:
        overall = (245, 245, 245)

    avg_noise = sum(side_noise) / len(side_noise) if side_noise else 99.0
    return medians, overall, avg_noise


def fill_background_patch(base: Image.Image, box, aggressive: bool = False):
    """원본 전체는 건드리지 않고 텍스트 박스만 주변 배경으로 덮습니다."""
    base = base.convert('RGB')
    x1, y1, x2, y2 = clamp_box(box, *base.size)
    medians, overall, noise = side_sample_stats(base, (x1, y1, x2, y2), margin=12)

    is_complex = noise > 34.0
    if is_complex and not aggressive:
        return base, False, noise, overall

    top = medians.get('top', overall)
    bottom = medians.get('bottom', overall)
    patch_h = max(1, y2-y1)
    patch_w = max(1, x2-x1)
    patch = Image.new('RGB', (patch_w, patch_h), overall)
    draw = ImageDraw.Draw(patch)
    for yy in range(patch_h):
        t = yy / max(1, patch_h-1)
        color = tuple(int(top[i]*(1-t) + bottom[i]*t) for i in range(3))
        draw.line((0, yy, patch_w, yy), fill=color)

    # 적극형에서는 약간의 블러를 섞어 경계가 너무 딱딱하지 않게 합니다.
    if aggressive and is_complex:
        patch = patch.filter(ImageFilter.GaussianBlur(radius=1.2))

    out = base.copy()
    out.paste(patch, (x1, y1))
    return out, True, noise, overall


def find_font_path(weight: str = 'regular'):
    bold = (weight or '').lower() == 'bold'
    candidates = [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc' if bold else '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    ]
    for path in candidates:
        try:
            with open(path, 'rb'):
                return path
        except Exception:
            pass
    return None


def fit_font(draw: ImageDraw.ImageDraw, text: str, box_w: int, box_h: int, weight: str = 'regular'):
    lines = max(1, len((text or '').splitlines()))
    max_size = max(8, min(120, int(box_h / max(1, lines) * 0.92)))
    min_size = 7
    path = find_font_path(weight)

    for size in range(max_size, min_size-1, -1):
        try:
            font = ImageFont.truetype(path, size=size) if path else ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
        spacing = max(1, int(size * 0.18))
        bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        if tw <= max(1, box_w) and th <= max(1, box_h):
            return font, spacing, size, tw, th

    font = ImageFont.truetype(path, size=min_size) if path else ImageFont.load_default()
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=1)
    return font, 1, min_size, bbox[2]-bbox[0], bbox[3]-bbox[1]


def ensure_text_contrast(text_rgb, bg_rgb):
    if abs(luminance(text_rgb) - luminance(bg_rgb)) >= 58:
        return text_rgb
    return (24, 24, 24) if luminance(bg_rgb) > 150 else (245, 245, 245)


def render_translation_image(original: Image.Image, segments, aggressive: bool = False):
    """
    추가 API 호출 없이 번역 결과에 포함된 위치정보로 텍스트만 교체합니다.
    복잡한 배경은 안전형에서 그대로 두고 검수 대상으로 표시합니다.
    """
    out = original.convert('RGB').copy()
    report = []

    # 위에서 아래 순서로 처리
    ordered = sorted(segments, key=lambda s: (s.get('bbox_px', [0, 10**9, 0, 0])[1], s.get('order', 9999)))
    for seg in ordered:
        box = seg.get('bbox_px')
        if not box or len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
            report.append({'order': seg.get('order'), 'status': 'skip', 'reason': '텍스트 위치를 확정하지 못함'})
            continue

        x1, y1, x2, y2 = clamp_box(box, *out.size)
        bw, bh = x2-x1, y2-y1
        if bw < 12 or bh < 8:
            report.append({'order': seg.get('order'), 'status': 'skip', 'reason': '텍스트 영역이 너무 작음'})
            continue

        if seg.get('background_complex', False) and not aggressive:
            report.append({'order': seg.get('order'), 'status': 'skip', 'reason': '사진/제품/복잡한 배경 위 문구라 안전형에서 보류'})
            continue

        # 지울 영역에 아주 작은 여백만 추가
        pad_x = max(2, min(8, int(bw * 0.025)))
        pad_y = max(2, min(6, int(bh * 0.08)))
        erase_box = clamp_box((x1-pad_x, y1-pad_y, x2+pad_x, y2+pad_y), *out.size)

        out, filled, noise, bg_rgb = fill_background_patch(out, erase_box, aggressive=aggressive)
        if not filled:
            report.append({'order': seg.get('order'), 'status': 'skip', 'reason': f'배경이 복잡함 (noise {noise:.1f})'})
            continue

        text = (seg.get('design_linebreak') or seg.get('translation') or '').strip()
        if not text:
            report.append({'order': seg.get('order'), 'status': 'skip', 'reason': '번역문 없음'})
            continue

        draw = ImageDraw.Draw(out)
        inner_pad = max(2, int(min(bw, bh) * 0.04))
        tx1, ty1, tx2, ty2 = x1+inner_pad, y1+inner_pad, x2-inner_pad, y2-inner_pad
        target_w = max(1, tx2-tx1)
        target_h = max(1, ty2-ty1)
        font, spacing, font_size, tw, th = fit_font(draw, text, target_w, target_h, seg.get('font_weight', 'regular'))

        if tw > target_w or th > target_h or font_size <= 7:
            report.append({'order': seg.get('order'), 'status': 'skip', 'reason': '번역문이 기존 영역에 안전하게 들어가지 않음'})
            continue

        align = seg.get('text_align', 'left')
        if align == 'center':
            x = tx1 + (target_w - tw) / 2
        elif align == 'right':
            x = tx2 - tw
        else:
            x = tx1
        y = ty1 + (target_h - th) / 2

        color = ensure_text_contrast(parse_hex_color(seg.get('text_color', '#222222')), bg_rgb)
        draw.multiline_text((x, y), text, font=font, fill=color, spacing=spacing, align=align)
        report.append({'order': seg.get('order'), 'status': 'replaced', 'reason': '', 'font_size': font_size})

    return out, report


def image_bytes(image: Image.Image, fmt: str = 'PNG') -> bytes:
    buf = io.BytesIO()
    if fmt.upper() == 'JPEG':
        image.convert('RGB').save(buf, format='JPEG', quality=95, optimize=True)
    else:
        image.save(buf, format='PNG', optimize=True)
    return buf.getvalue()

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
                        "design_linebreak": {"type": "string"},
                        "bbox": {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                            "minItems": 4,
                            "maxItems": 4
                        },
                        "text_color": {"type": "string"},
                        "text_align": {"type": "string", "enum": ["left", "center", "right"]},
                        "font_weight": {"type": "string", "enum": ["regular", "bold"]},
                        "background_complex": {"type": "boolean"},
                        "placement_review_required": {"type": "boolean"},
                        "placement_review_reason": {"type": "string"},
                        "type": {
                            "type": "string",
                            "enum": ["headline", "body", "label", "footnote", "test_value", "ingredients", "other"],
                        },
                        "review_required": {"type": "boolean"},
                        "review_reason": {"type": "string"},
                    },
                    "required": ["korean", "translation", "design_linebreak", "bbox", "text_color", "text_align", "font_weight", "background_complex", "placement_review_required", "placement_review_reason", "type", "review_required", "review_reason"],
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
14. translation에는 최종 번역문을 완성된 한 문장으로 적는다.
15. design_linebreak에는 translation의 단어/문자/문장부호를 절대 추가·삭제·변경하지 말고, 오직 실제 줄바꿈(\n)만 삽입한다.
16. 디자인 줄바꿈 기준:
    - headline/label: 가능하면 1줄, 길면 최대 2줄
    - body: 의미 단위가 자연스럽도록 2~3줄
    - footnote/test_value: 가능하면 1~2줄
    - 단어 중간에서 줄을 끊지 않는다.
    - 러시아어/영어/프랑스어/스페인어/베트남어는 전치사 하나만 줄 끝에 홀로 남지 않게 한다.
    - 일본어/중국어는 조사·구두점이 어색하게 다음 줄 첫 글자로 떨어지지 않게 한다.
17. 이미지 자동 교체를 위해 각 문구의 위치/스타일도 반환한다.
    - bbox=[x1,y1,x2,y2], 현재 입력 이미지 기준 0~1000 정규화 좌표. 해당 한국어 페이지 카피 글자 전체를 감싸되 제품/모델/아이콘은 포함하지 않는다.
    - text_color는 원래 한국어 글자의 대표 색상을 #RRGGBB로 추정한다.
    - text_align은 left/center/right 중 원래 배치에 가장 가까운 값.
    - font_weight는 regular/bold 중 선택.
    - background_complex는 글자 뒤가 사진, 모델, 제품, 복잡한 텍스처면 true. 단색/단순 그라데이션이면 false.
    - 위치가 애매하거나 번역문을 같은 영역에 넣기 어렵다면 placement_review_required=true와 이유를 적는다.
    - 위치를 전혀 확정할 수 없으면 bbox=[0,0,0,0]으로 반환한다.
18. 지정된 JSON 구조 외 설명을 쓰지 않는다.
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
            "bbox": [0, 0, 0, 0],
            "text_color": "#222222",
            "text_align": "left",
            "font_weight": "regular",
            "background_complex": True,
            "placement_review_required": True,
            "placement_review_reason": "재시도 응답에는 자동 배치 위치 정보가 없음",
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
        design_linebreak = str(seg.get("design_linebreak", "")).strip()
        bbox = seg.get("bbox", [0, 0, 0, 0])
        text_color = str(seg.get("text_color", "#222222")).strip() or "#222222"
        text_align = str(seg.get("text_align", "left")).strip() or "left"
        font_weight = str(seg.get("font_weight", "regular")).strip() or "regular"
        background_complex = bool(seg.get("background_complex", False))
        placement_review_required = bool(seg.get("placement_review_required", False))
        placement_review_reason = str(seg.get("placement_review_reason", "")).strip()
        seg_type = str(seg.get("type", "other")).strip() or "other"
        if not korean or not re.search(r"[가-힣]", korean):
            continue

        if korean in dictionary:
            translation = dictionary[korean]
            design_linebreak = ""

        if (not design_linebreak or normalize_linebreak_compare(design_linebreak) != normalize_linebreak_compare(translation)):
            design_linebreak = local_design_linebreak(translation, seg_type)

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

        if not valid_bbox_norm(bbox):
            bbox = [0, 0, 0, 0]
            placement_review_required = True
            placement_review_reason = placement_review_reason or "텍스트 위치 확인 필요"

        cleaned.append({
            "korean": korean,
            "translation": translation,
            "design_linebreak": design_linebreak,
            "bbox": [int(v) for v in bbox],
            "text_color": text_color,
            "text_align": text_align if text_align in {"left", "center", "right"} else "left",
            "font_weight": font_weight if font_weight in {"regular", "bold"} else "regular",
            "background_complex": background_complex,
            "placement_review_required": placement_review_required,
            "placement_review_reason": placement_review_reason,
            "type": seg_type,
            "review_required": bool(reasons),
            "review_reason": " / ".join(dict.fromkeys(reasons)),
            "length_ratio": ratio,
            "too_long": ratio > limit,
        })
    return cleaned


def deduplicate_segments(segments: List[Dict]) -> List[Dict]:
    best = {}
    sequence = []
    for idx, seg in enumerate(segments):
        seg['_seq'] = idx
        key = normalize_key(seg.get('korean', ''))
        if not key:
            continue
        if key not in best:
            best[key] = seg
            sequence.append(key)
            continue

        old = best[key]
        def score(item):
            box = item.get('bbox_px')
            valid = 1 if box and len(box) == 4 and box[2] > box[0] and box[3] > box[1] else 0
            not_review = 1 if not item.get('placement_review_required') else 0
            edge = float(item.get('_edge_score', 0))
            return (valid, not_review, edge)

        if score(seg) > score(old):
            seg['_seq'] = old.get('_seq', idx)
            best[key] = seg

    result = [best[k] for k in sequence]
    # 좌표가 있으면 실제 위→아래 순서를 우선, 없으면 원래 순서 유지
    result.sort(key=lambda s: (
        s.get('bbox_px', [0, 10**9, 0, 0])[1] if s.get('bbox_px') else 10**9,
        s.get('_seq', 10**9),
    ))
    for idx, seg in enumerate(result, start=1):
        seg['order'] = idx
        seg.pop('_seq', None)
        seg.pop('_edge_score', None)
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
        # 각 조각 내부 0~1000 좌표를 원본 이미지 픽셀 좌표로 변환
        cw, ch = chunk["image"].size
        for seg in segments:
            box = seg.get("bbox", [0, 0, 0, 0])
            if valid_bbox_norm(box):
                x1 = int(round(box[0] / 1000 * cw))
                y1_local = int(round(box[1] / 1000 * ch))
                x2 = int(round(box[2] / 1000 * cw))
                y2_local = int(round(box[3] / 1000 * ch))
                seg["bbox_px"] = [x1, chunk["y1"] + y1_local, x2, chunk["y1"] + y2_local]
                seg["_edge_score"] = min(box[0], box[1], 1000-box[2], 1000-box[3])
            else:
                seg["bbox_px"] = None
                seg["placement_review_required"] = True
                seg["placement_review_reason"] = seg.get("placement_review_reason") or "텍스트 위치 확인 필요"
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
            "[디자인 줄바꿈]",
            seg.get("design_linebreak", seg["translation"]),
            "",
            f"검수: {'필요' if seg.get('review_required') else '바로 사용'} {seg.get('review_reason', '')}",
            "",
        ])
    return "\n".join(parts)


def segments_to_csv(file_name: str, segments: List[Dict]) -> bytes:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["image", "order", "type", "korean", display_language, "design_linebreak", "review", "reason"])
    for seg in segments:
        writer.writerow([
            file_name,
            seg["order"],
            seg["type"],
            seg["korean"],
            seg["translation"],
            seg.get("design_linebreak", seg["translation"]),
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

        st.session_state["v122_results"] = results

# ------------------------------------------------------------
# 결과
# ------------------------------------------------------------
if st.session_state.get("v122_results"):
    results = st.session_state["v122_results"]
    st.markdown("---")

    total_segments = sum(len(r["segments"]) for r in results)
    review_count = sum(1 for r in results for s in r["segments"] if s.get("review_required"))
    failed_chunks = sum(1 for r in results for rep in r["reports"] if rep.get("status") == "failed")
    total_in = sum(r["input_tokens"] for r in results)
    total_out = sum(r["output_tokens"] for r in results)
    total_cost = sum(r["cost"] for r in results)
    placement_flag_count = sum(1 for r in results for s in r["segments"] if s.get("placement_review_required"))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("전체 문장", total_segments)
    c2.metric("⚠ 검수 필요", review_count)
    c3.metric("실패 조각", failed_chunks)
    c4.metric("이번 실행 예상 API 비용", f"${total_cost:.4f}")

    st.caption(f"이번 실행 토큰: 입력 {total_in:,} / 출력 {total_out:,} · 모델 {MODEL_ECONOMY} · 자동 배치 사전검수 {placement_flag_count}문장")

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

            preview_col, work_col = st.columns([0.72, 1.55], gap="large")

            with preview_col:
                st.markdown("#### 🖼️ 원본 이미지")
                st.image(
                    make_preview_image(r["image"]),
                    caption=f"{r['image'].width}×{r['image'].height}px",
                    use_container_width=True,
                )
                with st.expander("원본 크게 보기", expanded=False):
                    st.image(r["image"], use_container_width=True)

            edited_segments = []
            with work_col:
                st.markdown("#### 🌐 번역 + 디자인 줄바꿈")
                for seg in r["segments"]:
                    st.caption(f"{seg['order']}번 · {seg['type']}")

                    ko = st.text_area(
                        "한국어 원문",
                        value=seg["korean"],
                        key=f"ko_{r['index']}_{seg['order']}",
                        height=68,
                    )

                    tr = st.text_area(
                        display_language,
                        value=seg["translation"],
                        key=f"tr_{r['index']}_{seg['order']}",
                        height=74,
                    )

                    linebreak_value = st.text_area(
                        "✂️ 디자인 줄바꿈",
                        value=seg.get("design_linebreak", seg["translation"]),
                        key=f"lb_{r['index']}_{seg['order']}",
                        height=90,
                        help="번역 내용은 그대로 두고 실제 엔터만 넣은 디자인용 복사본입니다.",
                    )
                    line_count = max(1, len(linebreak_value.splitlines()))
                    st.caption(f"↳ {line_count}줄 권장 · 연결선/기호 없이 그대로 복사")

                    edited = dict(seg)
                    edited["korean"] = ko
                    edited["translation"] = tr
                    if normalize_linebreak_compare(linebreak_value) == normalize_linebreak_compare(tr):
                        edited["design_linebreak"] = linebreak_value
                    else:
                        edited["design_linebreak"] = local_design_linebreak(tr, seg.get("type", "other"))
                        st.warning("디자인 줄바꿈 칸에서 번역 내용이 변경되어, 번역문 기준으로 줄바꿈을 다시 맞췄습니다.")
                    edited_segments.append(edited)
                    st.markdown("---")

            st.markdown("#### 🖼️ 번역 이미지 만들기")
            st.caption("추가 API 호출 없이, 위 번역 결과의 위치정보를 이용해 원본 이미지의 한국어 영역만 교체합니다.")
            if st.button(
                "✨ 번역 이미지 생성 · 추가 API 비용 없음",
                key=f"make_img_{r['index']}",
                type="primary",
                use_container_width=True,
            ):
                aggressive = image_replace_mode == "적극형"
                rendered, placement_report = render_translation_image(
                    r["image"], edited_segments, aggressive=aggressive
                )
                st.session_state[f"v122_rendered_{r['index']}"] = {
                    "image": rendered,
                    "report": placement_report,
                }

            rendered_state = st.session_state.get(f"v122_rendered_{r['index']}")
            if rendered_state:
                rendered = rendered_state["image"]
                placement_report = rendered_state["report"]
                replaced_n = sum(1 for item in placement_report if item.get("status") == "replaced")
                skipped = [item for item in placement_report if item.get("status") != "replaced"]

                p1, p2 = st.columns(2)
                with p1:
                    st.markdown("**원본**")
                    st.image(make_preview_image(r["image"]), use_container_width=True)
                with p2:
                    st.markdown("**자동 번역 이미지**")
                    st.image(make_preview_image(rendered), use_container_width=True)

                if skipped:
                    st.warning(f"자동 교체 {replaced_n}개 · 배치 검수 {len(skipped)}개. 보류된 문구는 원본 한국어를 그대로 남겼습니다.")
                    with st.expander("배치 검수 내용 보기", expanded=False):
                        for item in skipped:
                            st.write(f"{item.get('order', '-')}번 · {item.get('reason', '')}")
                else:
                    st.success(f"✅ {replaced_n}개 문구 자동 교체 완료")

                png_bytes = image_bytes(rendered, 'PNG')
                jpg_bytes = image_bytes(rendered, 'JPEG')
                img_d1, img_d2 = st.columns(2)
                img_d1.download_button(
                    "📥 번역 이미지 PNG",
                    data=png_bytes,
                    file_name=f"{r['index']:02d}_{selected_market}_translated.png",
                    mime="image/png",
                    key=f"png_img_{r['index']}",
                    use_container_width=True,
                )
                img_d2.download_button(
                    "📥 번역 이미지 JPG",
                    data=jpg_bytes,
                    file_name=f"{r['index']:02d}_{selected_market}_translated.jpg",
                    mime="image/jpeg",
                    key=f"jpg_img_{r['index']}",
                    use_container_width=True,
                )

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
    st.info("이미지를 올리면 예상 분할 수와 API 호출 수를 먼저 보여줍니다. 번역 이미지 생성은 추가 API 비용 없이 로컬 처리합니다.")
