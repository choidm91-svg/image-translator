
import io
import zipfile
from PIL import Image, ImageDraw, ImageFont, ImageFilter

SIZE = 1024

def _font(size, bold=False):
    paths = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()

def _txt(draw, xy, s, size, bold=False, fill=(22,48,60), anchor=None):
    draw.multiline_text(xy, s, font=_font(size,bold), fill=fill, anchor=anchor, spacing=8)

def _bg(mode):
    accent_bg = (222,242,250) if mode == "Overseas" else (230,244,235)
    im = Image.new("RGBA",(SIZE,SIZE),"white")
    px=im.load()
    for y in range(SIZE):
        t=y/(SIZE-1)
        c=tuple(int(255*(1-t)+accent_bg[i]*t) for i in range(3))
        for x in range(SIZE):
            px[x,y]=(*c,255)
    return im

def _locked_product(product, max_w, max_h):
    """
    PRODUCT LOCK:
    - 원본 픽셀 내용 수정 금지
    - 생성형 보정 금지
    - 가로세로 비율 유지
    - 원본 전체를 동일 비율로 축소만 허용
    """
    p = product.copy().convert("RGBA")
    ratio = min(max_w/p.width, max_h/p.height, 1.0)
    if ratio < 1:
        p = p.resize((round(p.width*ratio), round(p.height*ratio)), Image.Resampling.LANCZOS)
    return p

def _place_product(base, product, center=(760,570), max_w=390, max_h=650):
    p=_locked_product(product,max_w,max_h)
    x=round(center[0]-p.width/2); y=round(center[1]-p.height/2)
    # 그림자는 제품 자체를 변경하지 않고 별도 레이어로만 생성
    alpha=p.getchannel("A")
    shadow=Image.new("RGBA",p.size,(0,0,0,0))
    sh=Image.new("RGBA",p.size,(30,50,60,60))
    sh.putalpha(alpha.filter(ImageFilter.GaussianBlur(14)))
    shadow.alpha_composite(sh)
    base.alpha_composite(shadow,(x+14,y+20))
    base.alpha_composite(p,(x,y))

def make_card(n, data, product, mode="Overseas"):
    accent=(13,113,177) if mode=="Overseas" else (31,112,76)
    base=_bg(mode); d=ImageDraw.Draw(base)
    _txt(d,(55,45),f"{n:02d}",25,True,accent)
    if n==1:
        _txt(d,(55,155),data["main_number"],118,True,accent)
        _txt(d,(58,290),data["main_claim"].upper(),44,True)
        _txt(d,(58,365),f'{data["ing1_name"].upper()} {data["ing1_value"]} + {data["ing2_name"].upper()} {data["ing2_value"]}',24,True,accent)
        _place_product(base,product,(760,590),390,650)
    elif n==2:
        _txt(d,(55,155),"SENSITIVE SKIN?",50,True)
        _txt(d,(55,225),"START WITH\nYOUR BARRIER.",58,True,accent)
        _txt(d,(55,770),"SOOTHE · HYDRATE · STRENGTHEN",26,True,accent)
        _place_product(base,product,(790,590),280,520)
    elif n==3:
        _txt(d,(512,245),f'{data["ing1_value"]} + {data["ing2_value"]}',80,True,accent,"mm")
        _txt(d,(512,380),f'= {data["main_number"]}',120,True,(20,48,60),"mm")
        _txt(d,(512,495),"ACTIVE FORMULA",34,True,accent,"mm")
    elif n==4:
        _txt(d,(55,165),data["ing1_name"].upper(),65,True,accent)
        _txt(d,(55,250),data["ing1_value"],120,True)
        _txt(d,(55,405),data["ing1_claim"].upper(),32,True,accent)
        _place_product(base,product,(780,600),270,510)
    elif n==5:
        _txt(d,(55,165),data["ing2_name"].upper(),65,True,accent)
        _txt(d,(55,250),data["ing2_value"],120,True)
        _txt(d,(55,405),data["ing2_claim"].upper(),32,True,accent)
        d.ellipse((600,320,880,600),fill=(242,252,255),outline=accent,width=5)
        _txt(d,(740,460),"B5",76,True,accent,"mm")
    elif n==6:
        _txt(d,(512,160),"3 KEY BENEFITS",42,True,accent,"mm")
        for y,b in zip((340,515,690),data["benefits"]):
            d.rounded_rectangle((120,y-55,904,y+65),35,fill=(255,255,255,220),outline=accent,width=3)
            _txt(d,(512,y),b.upper(),38,True,(20,48,60),"mm")
    elif n==7:
        _txt(d,(55,160),"ONE DROP.",58,True,accent)
        _txt(d,(55,235),"LIGHTWEIGHT\nHYDRATION.",58,True)
        d.ellipse((585,300,855,570),fill=(245,253,255),outline=accent,width=5)
        _txt(d,(720,435),"DROP",38,True,accent,"mm")
        _txt(d,(55,780),data["texture"].upper(),27,True,accent)
    elif n==8:
        base=Image.new("RGBA",(SIZE,SIZE),"white"); d=ImageDraw.Draw(base)
        _txt(d,(512,160),"DERMATOLOGICALLY TESTED",29,True,accent,"mm")
        _txt(d,(512,365),data["test_value"],150,True,accent,"mm")
        _txt(d,(512,490),data["test_label"].upper(),34,True,(20,48,60),"mm")
        _txt(d,(512,700),"TEST COMPLETED",25,True,(75,95,105),"mm")
    else:
        _txt(d,(55,165),"YOUR DAILY",52,True,accent)
        _txt(d,(55,230),"BARRIER AMPOULE",52,True)
        _txt(d,(55,760),f'{data["ing1_name"].upper()} {data["ing1_value"]} · {data["ing2_name"].upper()} {data["ing2_value"]}',25,True,accent)
        _txt(d,(55,815),data["volume"],22,True,(75,95,105))
        _place_product(base,product,(765,590),390,650)
    return base.convert("RGB")

def make_zip(cards):
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as z:
        for i,im in enumerate(cards,1):
            b=io.BytesIO(); im.save(b,"PNG")
            z.writestr(f"{i:02d}.png",b.getvalue())
    return buf.getvalue()
