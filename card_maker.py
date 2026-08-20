import io, zipfile
from PIL import Image, ImageDraw, ImageFont, ImageFilter
SIZE=1024
def font(sz,b=False):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if b else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        try:return ImageFont.truetype(p,sz)
        except:pass
    return ImageFont.load_default()
def locked_product(im,maxw,maxh):
    p=im.convert("RGBA").copy()
    r=min(maxw/p.width,maxh/p.height,1.0)
    if r<1:p=p.resize((round(p.width*r),round(p.height*r)),Image.Resampling.LANCZOS)
    return p
def place(base,product,cx,cy,maxw,maxh):
    p=locked_product(product,maxw,maxh); x=int(cx-p.width/2); y=int(cy-p.height/2)
    base.alpha_composite(p,(x,y))
def card(n,d,product,mode):
    accent=(13,113,177) if mode=="해외형" else (31,112,76)
    base=Image.new("RGBA",(SIZE,SIZE),(246,251,253,255)); dr=ImageDraw.Draw(base)
    def t(x,y,s,sz,b=False,c=(25,48,58),a=None):dr.multiline_text((x,y),s,font=font(sz,b),fill=c,anchor=a,spacing=8)
    t(55,45,f"{n:02d}",24,True,accent)
    if n==1:
        t(55,165,d["main_number"],120,True,accent); t(55,300,d["main_claim"].upper(),44,True)
        t(55,375,f'{d["ing1"]} {d["v1"]} + {d["ing2"]} {d["v2"]}',25,True,accent); place(base,product,780,620,360,620)
    elif n==2:
        t(55,180,"SENSITIVE SKIN?",52,True); t(55,250,"START WITH\nYOUR BARRIER.",58,True,accent); place(base,product,790,650,280,500)
    elif n==3:
        t(512,260,f'{d["v1"]} + {d["v2"]}',82,True,accent,"mm"); t(512,400,f'= {d["main_number"]}',120,True,(25,48,58),"mm"); t(512,520,"ACTIVE FORMULA",34,True,accent,"mm")
    elif n==4:
        t(55,190,d["ing1"].upper(),68,True,accent); t(55,285,d["v1"],120,True); t(55,440,d["claim1"].upper(),30,True,accent); place(base,product,790,650,270,500)
    elif n==5:
        t(55,190,d["ing2"].upper(),68,True,accent); t(55,285,d["v2"],120,True); t(55,440,d["claim2"].upper(),30,True,accent)
    elif n==6:
        t(512,170,"3 KEY BENEFITS",42,True,accent,"mm")
        for y,s in zip([350,520,690],d["benefits"]):
            dr.rounded_rectangle((130,y-55,894,y+65),35,fill="white",outline=accent,width=3); t(512,y,s.upper(),38,True,(25,48,58),"mm")
    elif n==7:
        t(55,180,"ONE DROP.",58,True,accent); t(55,255,"LIGHTWEIGHT\nHYDRATION.",58,True); t(55,790,d["texture"].upper(),27,True,accent)
    elif n==8:
        base=Image.new("RGBA",(SIZE,SIZE),"white");dr=ImageDraw.Draw(base)
        t=lambda x,y,s,sz,b=False,c=(25,48,58),a=None:dr.multiline_text((x,y),s,font=font(sz,b),fill=c,anchor=a,spacing=8)
        t(512,170,"DERMATOLOGICALLY TESTED",30,True,accent,"mm");t(512,390,d["test"],150,True,accent,"mm");t(512,520,d["test_label"].upper(),34,True,(25,48,58),"mm")
    else:
        t(55,190,"YOUR DAILY",52,True,accent);t(55,255,"BARRIER AMPOULE",52,True);place(base,product,770,620,380,620)
    return base.convert("RGB")
def zip_cards(cards):
    b=io.BytesIO()
    with zipfile.ZipFile(b,"w",zipfile.ZIP_DEFLATED) as z:
        for i,im in enumerate(cards,1):
            x=io.BytesIO();im.save(x,"PNG");z.writestr(f"{i:02d}.png",x.getvalue())
    return b.getvalue()
