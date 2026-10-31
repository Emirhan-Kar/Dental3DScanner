"""
DentalScanner demo arayuzu (yerel, internetsiz calisir).

Calistirma:   venv_gpu\\Scripts\\python.exe demo\\app.py      (ya da demo_baslat.bat)
Tarayicida:   http://127.0.0.1:8050
"""
import os
import sys
import base64
import tempfile
import threading
import webbrowser
import numpy as np

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

import dash
from dash import dcc, html, Input, Output, State, ctx, no_update, Patch
import plotly.graph_objects as go
from flask import send_from_directory

from demo.motor import (TaramaMotoru, vakalari_listele, ASAMALAR, PALET, cihaz, hasta_ciftleri,
                        kapanis_taramalari, tarama_gosterimi, orijinal_gosterimi, kapanis_konumu,
                        olcum_paketi, capraskilik_yorumu)
from processing.klinik_olcum import ark_sirasi, bolton, BOLTON_ON_IDEAL, BOLTON_TOPLAM_IDEAL

motor = TaramaMotoru()
VAKALAR = vakalari_listele()
CIFTLER = hasta_ciftleri()
GORSEL_DIZIN = os.path.join(KOK, "rapor_gorselleri")

UST = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
ALT = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]
FDI_IDX = {f: i for i, f in enumerate([0] + [q * 10 + k for q in (1, 2, 3, 4) for k in range(1, 9)])}

app = dash.Dash(__name__, title="DentalScanner — Demo", update_title=None, suppress_callback_exceptions=True)
server = app.server


@server.route("/gorsel/<path:ad>")
def gorsel(ad):
    return send_from_directory(GORSEL_DIZIN, ad)


# ───────────────────────── yardimcilar ─────────────────────────

def _rgb(c):
    return f"rgb({int(c[0] * 255)},{int(c[1] * 255)},{int(c[2] * 255)})"


def _sahne(asagidan=False):
    """Dislerin okluzal yuzunu gosteren kamera: alt cene ustten, ust cene alttan bakilir."""
    z = -1.6 if asagidan else 1.6
    return dict(
        xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
        aspectmode="data", bgcolor="#F4F6F3",
        camera=dict(eye=dict(x=0, y=-0.15, z=z), up=dict(x=0, y=1, z=0)),
    )


def _ust_cene_mi(V, etiket):
    """Disler dis etinin altindaysa (z ekseninde) ust cenedir."""
    e = np.asarray(etiket)
    if (e > 0).sum() < 50 or (e == 0).sum() < 50:
        return False
    return V[e > 0, 2].mean() < V[e == 0, 2].mean()


def bos_sekil(mesaj=""):
    fig = go.Figure()
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), plot_bgcolor="#F4F6F3",
                      margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="#F4F6F3",
                      annotations=[dict(text=mesaj, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper",
                                        font=dict(size=16, color="#6B7680"))] if mesaj else [])
    return fig


def nokta_sekli(P):
    if P is None or len(P) == 0:
        return bos_sekil("Tarama başladığında 3D noktalar burada birikecek")
    fig = go.Figure(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers",
                                 marker=dict(size=1.3, color=P[:, 2], colorscale=[[0, "#8A9AA8"], [1, "#1B2430"]], opacity=0.9),
                                 hoverinfo="skip"))
    fig.update_layout(scene=_sahne(), margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="#F4F6F3", uirevision="nokta")
    return fig


def yuzey_sekli(y, vmax=0.1, asagidan=False):
    V, F, s = y["V"], y["F"], y["sapma"]
    fig = go.Figure(go.Mesh3d(x=V[:, 0], y=V[:, 1], z=V[:, 2], i=F[:, 0], j=F[:, 1], k=F[:, 2],
                              intensity=np.clip(s, 0, vmax), colorscale="Viridis", cmin=0, cmax=vmax,
                              colorbar=dict(title=dict(text="Sapma (mm)", side="right"), thickness=12, len=0.6),
                              lighting=dict(ambient=0.55, diffuse=0.7, specular=0.15, roughness=0.6),
                              hovertemplate="sapma %{intensity:.3f} mm<extra></extra>"))
    fig.update_layout(scene=_sahne(asagidan), margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="#F4F6F3", uirevision="yuzey")
    return fig


def seg_sekli(s):
    V, F, e = s["V"], s["F"], np.asarray(s["etiket"]).astype(int)
    renk = [_rgb(PALET[int(c)]) for c in e]
    fdi = [0] + [q * 10 + k for q in (1, 2, 3, 4) for k in range(1, 9)]
    metin = ["diş eti" if c == 0 else f"diş {fdi[c]}" for c in e]
    fig = go.Figure(go.Mesh3d(x=V[:, 0], y=V[:, 1], z=V[:, 2], i=F[:, 0], j=F[:, 1], k=F[:, 2],
                              vertexcolor=renk, text=metin, hovertemplate="%{text}<extra></extra>",
                              lighting=dict(ambient=0.6, diffuse=0.7, specular=0.1, roughness=0.7)))
    fig.update_layout(scene=_sahne(_ust_cene_mi(V, e)), margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="#F4F6F3", uirevision="seg")
    return fig


def odontogram(disler=None, iou=None):
    """FDI dis semasi: ust cene ustte, alt cene altta; hastanin sagi ekranin solunda."""
    disler = set(disler or [])
    iou = iou or {}

    def hucre(f):
        bulundu = f in disler
        stil = {}
        if bulundu:
            c = PALET[FDI_IDX[f]]
            stil = {"background": _rgb(c), "color": "#10161D" if sum(c) > 1.4 else "#FFFFFF"}
        ipucu = f"Diş {f}" + (f", IoU {iou[f]:.2f}" if f in iou else "") + ("" if bulundu else " (bulunamadı)")
        return html.Div([html.Span(str(f), className="fdi"),
                         html.Span(f"{iou[f]:.2f}" if (bulundu and f in iou) else "", className="iou")],
                        className="dis" + (" bulundu" if bulundu else ""), style=stil, title=ipucu)

    def cene(liste, ad):
        return html.Div([html.Div([hucre(f) for f in liste[:8]], className="yari"),
                         html.Div([hucre(f) for f in liste[8:]], className="yari")], className="cene", title=ad)

    return html.Div([
        html.Div([html.Span("Hastanın sağı"), html.Span("Hastanın solu")], className="yonler"),
        cene(UST, "Üst çene"),
        html.Div(className="orta-cizgi"),
        cene(ALT, "Alt çene"),
    ], className="odontogram")


def _mm(x, n=1):
    return "–" if x is None else f"{x:.{n}f}".replace(".", ",").replace("-", "−")


def olcum_cizgileri(gorsel):
    """3D sahneye eklenecek izler: ark egrisi ve her disin genislik cizgisi."""
    izler = []
    if gorsel is None:
        return izler
    if gorsel.get("egri") is not None:
        E = np.asarray(gorsel["egri"])
        izler.append(go.Scatter3d(x=E[:, 0], y=E[:, 1], z=E[:, 2], mode="lines", line=dict(color="#D7322B", width=6),
                                  name="Ark eğrisi", hoverinfo="name"))
    xs, ys, zs, mx, my, mz, mt = [], [], [], [], [], [], []
    for f, (a, b) in gorsel["cizgiler"].items():
        xs += [a[0], b[0], None]; ys += [a[1], b[1], None]; zs += [a[2], b[2], None]
        m = (np.asarray(a) + np.asarray(b)) / 2
        mx.append(m[0]); my.append(m[1]); mz.append(m[2] + 1.2)
        mt.append(f"{f}: {np.linalg.norm(np.asarray(a) - np.asarray(b)):.1f}".replace(".", ","))
    izler.append(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color="#10161D", width=5),
                              name="Meziodistal genişlik", hoverinfo="skip"))
    izler.append(go.Scatter3d(x=mx, y=my, z=mz, mode="text", text=mt, textfont=dict(size=11, color="#10161D"),
                              hoverinfo="skip", name="Genişlik (mm)"))
    return izler


def olcum_sekli(s, gorsel):
    fig = seg_sekli(s)
    for iz in olcum_cizgileri(gorsel):
        fig.add_trace(iz)
    fig.update_traces(selector=dict(type="mesh3d"), opacity=0.55)
    fig.update_layout(showlegend=False, uirevision="olcum")
    return fig


def olcum_paneli(tahmin, gercek=None, baslik="Klinik ölçümler (otomatik)"):
    """Ozet kartlari + dis genislik tablosu. tahmin/gercek: processing.klinik_olcum.ozet ciktisi."""
    if not tahmin or not tahmin.get("genislik"):
        return html.P("Ölçüm için yeterli diş bulunamadı.", className="bos-durum")
    g = gercek or {}

    def kart(ad, anahtar, aciklama=None):
        t = tahmin.get(anahtar)
        alt = []
        if gercek is not None:
            alt.append(html.Span(f"gerçek: {_mm(g.get(anahtar))} mm", className="gercek"))
        if aciklama:
            alt.append(html.Span(aciklama, className="yorum"))
        return html.Div([html.Div([html.Span(_mm(t), className="deger"), html.Span("mm", className="birim")]),
                         html.Div(ad, className="aciklama"), html.Div(alt, className="alt-satir")], className="metrik")

    kartlar = [
        kart("Ark uzunluğu farkı (gerekli − mevcut)", "ark_farki", capraskilik_yorumu(tahmin.get("ark_farki"))),
        kart("Mevcut ark uzunluğu (6–6)", "ark_mevcut"),
        kart("Gerekli alan (5–5 diş genişlikleri)", "ark_gerekli"),
        kart("İnterkanin genişlik", "interkanin"),
        kart("İntermolar genişlik", "intermolar"),
    ]
    sira = [f for f in ark_sirasi(tahmin["cene"]) if f in tahmin["genislik"] or f in g.get("genislik", {})]
    farklar = []
    bas = [html.Th("Diş")] + [html.Th(str(f)) for f in sira]
    satir_t = [html.Th("Yapay zekâ")] + [html.Td(_mm(tahmin["genislik"].get(f))) for f in sira]
    satirlar = [html.Tr(bas), html.Tr(satir_t)]
    if gercek is not None:
        satirlar.append(html.Tr([html.Th("Gerçek")] + [html.Td(_mm(g["genislik"].get(f))) for f in sira]))
        hucre = [html.Th("Fark")]
        for f in sira:
            a, b = g["genislik"].get(f), tahmin["genislik"].get(f)
            if a is None or b is None:
                hucre.append(html.Td("yok" if b is None else "fazla", className="kotu"))
                continue
            d = b - a
            farklar.append(abs(d))
            hucre.append(html.Td("0,0" if abs(d) < 0.05 else ("+" if d > 0 else "") + _mm(d), className="iyi" if abs(d) <= 0.5 else "orta" if abs(d) <= 1 else "kotu"))
        satirlar.append(html.Tr(hucre, className="fark-satir"))
    if farklar:
        kartlar.append(html.Div([html.Div([html.Span(_mm(float(np.mean(farklar)), 2), className="deger"), html.Span("mm", className="birim")]),
                                 html.Div("Diş genişliği ortalama mutlak hata", className="aciklama"),
                                 html.Div(html.Span(f"{sum(x <= 0.5 for x in farklar)}/{len(farklar)} diş 0,5 mm içinde", className="yorum"), className="alt-satir")],
                                className="metrik"))
    notlar = [html.P("Genişlik: dişin ark boyunca en geniş meziodistal boyu. Gerekli alan > mevcut ark ise fark "
                     "çapraşıklıktır (dişlerin sığmadığı alan), negatifse dişler arasında boşluk vardır.", className="not")]
    if tahmin.get("uyari"):
        notlar.append(html.P(tahmin["uyari"], className="not uyari"))
    return html.Div([html.H2(baslik), html.Div(kartlar, className="olcum-kartlar"),
                     html.Div(html.Table(satirlar, className="olcum-tablo"), className="tablo-kaydir")] + notlar)


YER_TUTUCU = ("data:image/svg+xml;utf8," +
              "<svg xmlns='http://www.w3.org/2000/svg' width='1760' height='480'>"
              "<rect width='100%25' height='100%25' fill='%230A0D10'/>"
              "<text x='50%25' y='50%25' fill='%237C8790' font-family='Segoe UI,Arial' font-size='26' text-anchor='middle'>"
              "Taramayı başlatınca kamera görüntüsü, tespit edilen lazer çizgisi ve probun konumu burada akar</text></svg>")


def asama_seridi(aktif, bitti=False):
    sira = [a for a, _ in ASAMALAR if a != "bitti"]
    idx = sira.index(aktif) if aktif in sira else (len(sira) if aktif == "bitti" else -1)
    ogeler = []
    for n, (a, ad) in enumerate([x for x in ASAMALAR if x[0] != "bitti"]):
        durum = "tamam" if n < idx else ("aktif" if n == idx else "")
        ogeler.append(html.Li([html.Span(str(n + 1), className="no"), html.Span(ad)], className=durum))
    return html.Ol(ogeler, className="asamalar")


def metrik(deger, birim, aciklama):
    return html.Div([html.Div([html.Span(deger, className="deger"), html.Span(birim, className="birim")]),
                     html.Div(aciklama, className="aciklama")], className="metrik")


# ───────────────────────── yerlesim ─────────────────────────

_varsayilan = next((v["yol"] for v in VAKALAR if v["ad"] == "VBUD4E5R_lower"), VAKALAR[0]["yol"] if VAKALAR else None)
vaka_secenek = [{"label": f"{v['ad']}" + (" (test seti, model görmedi)" if v["test"] else " (eğitim seti)"),
                 "value": v["yol"]} for v in VAKALAR]

kontrol = html.Aside([
    html.Label("Hasta modeli", htmlFor="vaka"),
    dcc.Dropdown(id="vaka", options=vaka_secenek, value=_varsayilan,
                 clearable=False),
    html.Label("Tarama geçişleri"),
    dcc.Checklist(id="gecisler", options=[{"label": " Tepeden", "value": "tepe"},
                                          {"label": " Yanaktan (35°)", "value": "yanak"},
                                          {"label": " Dilden (−35°)", "value": "dil"}],
                  value=["tepe", "yanak", "dil"], className="secenekler"),
    html.Label("Kareler arası mesafe"),
    dcc.RadioItems(id="adim", options=[{"label": " 0,25 mm (en iyi kalite, en yavaş)", "value": 0.25},
                                       {"label": " 0,5 mm (dengeli)", "value": 0.5},
                                       {"label": " 1 mm (hızlı, seyrek)", "value": 1.0}],
                   value=0.5, className="secenekler"),
    html.Label("Probun konumu"),
    dcc.RadioItems(id="mod", options=[{"label": " Biliniyor (ideal)", "value": "ideal"},
                                      {"label": " Sadece IMU (kaymalı)", "value": "imu_ham"},
                                      {"label": " IMU + hibrit ICP düzeltmesi", "value": "imu"}],
                   value="ideal", className="secenekler"),
    html.Label("Kamera gürültüsü"),
    dcc.Slider(id="gurultu", min=0, max=25, step=1, value=3, marks={0: "0", 3: "3", 10: "10", 25: "25"}),
    html.Div([
        html.Button("Taramayı başlat", id="baslat", className="birincil"),
        html.Button("Durdur", id="durdur", className="ikincil"),
    ], className="dugmeler"),
    dcc.Checklist(id="kayit", options=[{"label": " Kayıtlı tarama varsa hızlı oynat", "value": "evet"}],
                  value=["evet"], className="secenekler kucuk"),
    html.Pre(id="gunluk", className="gunluk"),
], className="kontrol")

canli = html.Section([
    kontrol,
    html.Div([
        html.Div([
            html.Div([
                html.H2("Sanal kamera"),
                html.Span(id="kare-bilgi", className="kare-bilgi"),
            ], className="baslik-satir"),
            html.Div([html.Span("Kameranın gördüğü (parlaklığı artırıldı)"),
                      html.Span("Yazılımın bulduğu lazer çizgisi"),
                      html.Span("Probun ağızdaki konumu (üstten)")], className="kamera-etiketler"),
            html.Img(id="kamera", src=YER_TUTUCU, className="kamera",
                     alt="Solda kamera görüntüsü, ortada tespit edilen lazer noktaları, sağda probun çene üzerindeki konumu"),
            html.P("Prob dişlerin 17 mm üzerinden ark boyunca ilerliyor. Kırmızı çizgi dişe düşen lazer; yeşil noktalar "
                   "yazılımın her piksel sütununda bulduğu çizgi merkezi. Her kare çenenin tek bir kesitini verir, "
                   "haritadaki mavi alan şimdiye kadar taranan bölge.", className="kamera-aciklama"),
            html.Div(html.Div(id="ilerleme-dolu", className="dolu"), className="ilerleme"),
            html.Div(id="asamalar"),
        ], className="panel kamera-panel"),
        html.Div([
            html.Div([
                html.H2(id="sahne-baslik", children="3D model"),
                dcc.RadioItems(id="gorunum", options=[{"label": "Nokta bulutu", "value": "nokta"},
                                                      {"label": "Sapma haritası", "value": "yuzey"},
                                                      {"label": "Diş segmentasyonu", "value": "seg"},
                                                      {"label": "Ölçümler", "value": "olcum"}],
                               value="nokta", className="sekme-secici", inline=True),
            ], className="baslik-satir"),
            dcc.Graph(id="sahne", figure=bos_sekil("Tarama başladığında 3D noktalar burada birikecek"),
                      config={"displaylogo": False, "modeBarButtonsToRemove": ["toImage"]}, className="sahne"),
        ], className="panel sahne-panel"),
        html.Div([
            html.Div(id="odontogram", children=odontogram()),
            html.Div(id="metrikler", className="metrikler",
                     children=html.P("Tarama bitince doğruluk ve segmentasyon ölçümleri burada görünür. "
                                     "Diş şemasındaki her kutu, model o dişi bulduğunda kendi rengini alır.", className="bos-durum")),
        ], className="panel alt-panel"),
        html.Div(id="olcum-panel", className="panel olcum-panel",
                 children=html.P("Tarama bitince yapay zekânın segmentasyonundan diş genişlikleri, ark uzunluğu ve "
                                 "çapraşıklık otomatik ölçülür; uzman etiketlerinden çıkan değerlerle yan yana gösterilir.",
                                 className="bos-durum")),
    ], className="calisma"),
], className="canli")

yukle = html.Section([
    html.Div([
        html.H2("Kendi dosyanızı segmentleyin"),
        html.P("Bir ağız içi tarama dosyası (.obj, .ply veya .stl) yükleyin. Model her dişi FDI numarasıyla işaretler."),
        dcc.Upload(id="yukleme", children=html.Div(["Dosyayı buraya sürükleyin ya da ", html.U("seçin")]),
                   className="yukleme", multiple=False, max_size=200 * 1024 * 1024),
        dcc.Loading(html.Div(id="yukleme-durum", className="yukleme-durum"), type="circle", color="#2E5E7E"),
        html.Div(id="yukleme-odontogram", children=odontogram()),
        html.Div(id="yukleme-olcum", className="yukleme-olcum"),
    ], className="panel yukle-sol"),
    html.Div(dcc.Graph(id="yukleme-sahne", figure=bos_sekil("Yüklenen model burada görünecek"),
                       config={"displaylogo": False}, className="sahne"), className="panel yukle-sag"),
], className="yukle")


# ───────────────────────── iki cene birlikte ─────────────────────────

def _cift_secenekleri():
    hazir = {h for h in CIFTLER if kapanis_taramalari(h)}
    sira = sorted(CIFTLER, key=lambda h: (h not in hazir, not CIFTLER[h]["test"], h))
    return [{"label": h + (" (iki çene de tarandı)" if h in hazir else "") + (" · test seti" if CIFTLER[h]["test"] else ""),
             "value": h} for h in sira]


_KAPANIS_ONBELLEK = {}


def _kapanis_verisi(hasta, kaynak, tarama_no):
    """(alt, ust, R, t, bilgi) — ayni secim icin bellekte tutulur."""
    anahtar = (hasta, kaynak, tarama_no)
    if anahtar in _KAPANIS_ONBELLEK:
        return _KAPANIS_ONBELLEK[anahtar]
    if kaynak == "tarama":
        liste = kapanis_taramalari(hasta)
        if not liste:
            return None
        s = liste[min(int(tarama_no or 0), len(liste) - 1)]
        alt, ust, bilgi = tarama_gosterimi(s["lower"]), tarama_gosterimi(s["upper"]), s["etiket"]
    else:
        alt = orijinal_gosterimi(CIFTLER[hasta]["lower"], motor)
        ust = orijinal_gosterimi(CIFTLER[hasta]["upper"], motor)
        bilgi = "veri setindeki orijinal modeller"
    R, t = kapanis_konumu(alt, ust)
    for m, cene in ((alt, "alt"), (ust, "ust")):
        try:
            m["olcum"] = olcum_paketi(m["V"], m["F"], m["etiket"], cene=cene)["ham"]
        except Exception:
            m["olcum"] = None
    if len(_KAPANIS_ONBELLEK) > 6:
        _KAPANIS_ONBELLEK.clear()
    _KAPANIS_ONBELLEK[anahtar] = (alt, ust, R, t, bilgi)
    return _KAPANIS_ONBELLEK[anahtar]


def _cene_izi(m, V, renk, ad, vmax):
    F = m["F"]
    ortak = dict(x=V[:, 0], y=V[:, 1], z=V[:, 2], i=F[:, 0], j=F[:, 1], k=F[:, 2], name=ad,
                 lighting=dict(ambient=0.55, diffuse=0.75, specular=0.12, roughness=0.65),
                 lightposition=dict(x=0, y=-200, z=100))
    if renk == "sapma" and m["sapma"] is not None:
        return go.Mesh3d(intensity=np.clip(m["sapma"], 0, vmax), colorscale="Viridis", cmin=0, cmax=vmax,
                         showscale=(ad == "Alt çene"),
                         colorbar=dict(title=dict(text="Sapma (mm)", side="right"), thickness=12, len=0.6),
                         hovertemplate=ad + ", sapma %{intensity:.3f} mm<extra></extra>", **ortak)
    e = m["etiket"]
    fdi = [0] + [q * 10 + k for q in (1, 2, 3, 4) for k in range(1, 9)]
    return go.Mesh3d(vertexcolor=[_rgb(PALET[int(c)]) for c in e],
                     text=[f"{ad}, diş eti" if c == 0 else f"{ad}, diş {fdi[c]}" for c in e],
                     hovertemplate="%{text}<extra></extra>", **ortak)


def kapanis_sekli(alt, ust, R, t, aciklik, renk, vmax):
    Vu = ust["V"] @ R.T + t + np.array([0, 0, aciklik])
    fig = go.Figure([_cene_izi(alt, alt["V"], renk, "Alt çene", vmax), _cene_izi(ust, Vu, renk, "Üst çene", vmax)])
    fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                                 aspectmode="data", bgcolor="#F4F6F3",
                                 camera=dict(eye=dict(x=0.7, y=-1.35, z=0.28), up=dict(x=0, y=0, z=1))),
                      margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="#F4F6F3", showlegend=False, uirevision="kapanis")
    return fig


def _cene_metrikleri(ad, m):
    ogeler = [metrik(str(len(m["disler"])), "diş", f"{ad}: bulunan diş")]
    if m["mIoU"] is not None:
        ogeler.append(metrik(f"{m['mIoU']:.3f}".replace(".", ","), "", f"{ad}: segmentasyon mIoU"))
    if m["rms"] is not None:
        ogeler.append(metrik(f"{m['rms']:.3f}".replace(".", ","), "mm", f"{ad}: yüzey sapması (RMS)"))
    o = m.get("olcum")
    fark = o["ark"]["fark"] if (o and o["ark"]) else None
    ogeler.append(metrik(_mm(fark), "mm", f"{ad}: ark uzunluğu farkı ({capraskilik_yorumu(fark)})"))
    return ogeler


def _bolton_kartlari(alt, ust):
    if not (alt.get("olcum") and ust.get("olcum")):
        return []
    b = bolton(alt["olcum"], ust["olcum"])
    kartlar = []
    for ad, anahtar, (ideal, ss) in (("ön (3–3)", "on", BOLTON_ON_IDEAL), ("toplam (6–6)", "toplam", BOLTON_TOPLAM_IDEAL)):
        x = b[anahtar]
        if x is None:
            yorum = "eksik diş nedeniyle hesaplanamadı"
        elif x > ideal + ss:
            yorum = "alt çene dişleri görece büyük"
        elif x < ideal - ss:
            yorum = "üst çene dişleri görece büyük"
        else:
            yorum = "normal aralıkta"
        kartlar.append(html.Div([html.Div([html.Span(_mm(x), className="deger"), html.Span("%", className="birim")]),
                                 html.Div(f"Bolton {ad} oranı", className="aciklama"),
                                 html.Div([html.Span(f"ideal {_mm(ideal)} ± {_mm(ss, 2)}", className="gercek"),
                                           html.Span(yorum, className="yorum")], className="alt-satir")],
                                className="metrik"))
    return kartlar


_ilk_hasta = "VBUD4E5R" if "VBUD4E5R" in CIFTLER else (next(iter(CIFTLER)) if CIFTLER else None)

kapanis = html.Section([
    html.Aside([
        html.Label("Hasta", htmlFor="k-hasta"),
        dcc.Dropdown(id="k-hasta", options=_cift_secenekleri(), value=_ilk_hasta, clearable=False),
        html.Label("Gösterilecek model"),
        dcc.RadioItems(id="k-kaynak", options=[{"label": " Sanal taramadan elde edilen", "value": "tarama"},
                                               {"label": " Veri setindeki orijinal", "value": "orijinal"}],
                       value="tarama", className="secenekler"),
        html.Div([html.Label("Tarama ayarı", htmlFor="k-tarama"),
                  dcc.Dropdown(id="k-tarama", options=[], value=0, clearable=False)], id="k-tarama-kutu"),
        html.Label("Renklendirme"),
        dcc.RadioItems(id="k-renk", options=[{"label": " Diş segmentasyonu", "value": "seg"},
                                             {"label": " Sapma haritası", "value": "sapma"}],
                       value="seg", className="secenekler"),
        html.Label("Çeneleri aç"),
        dcc.Slider(id="k-aciklik", min=0, max=30, step=1, value=8, updatemode="drag",
                   marks={0: "kapalı", 10: "10 mm", 20: "20", 30: "30 mm"}),
        html.P("Veri setinde hastanın ısırma kaydı olmadığı için iki çene, diş tepeleri birbirine oturacak şekilde "
               "yaklaşık konumlandırıldı. Çeneler ayrı ayrı taranıp ayrı ayrı segmentlendi.", className="not"),
        dcc.Loading(html.Div(id="k-durum", className="yukleme-durum"), type="circle", color="#2E5E7E"),
    ], className="kontrol"),
    html.Div([
        html.Div([html.H2("İki çene birlikte"), html.Span(id="k-bilgi", className="kare-bilgi")], className="baslik-satir"),
        dcc.Graph(id="k-sahne", figure=bos_sekil("Bir hasta seçin"), config={"displaylogo": False}, className="sahne kapanis-sahne"),
    ], className="panel kapanis-sahne-panel"),
    html.Div([
        html.Div(id="k-odontogram", children=odontogram()),
        html.Div(id="k-metrikler", className="metrikler"),
    ], className="panel kapanis-alt"),
], className="kapanis")


SEKIL_ADLARI = {
    1: "Segmentasyon: üç modelin test seti karşılaştırması",
    2: "Segmentasyon: öğrenme eğrileri",
    3: "Segmentasyon: diş bazında IoU",
    4: "Segmentasyon: karışıklık matrisi (DGCNN)",
    5: "Segmentasyon: tarama başına başarı dağılımı",
    6: "Segmentasyon: en iyi ve en kötü örnekler",
    7: "Birleştirme: parça çifti hizalama başarısı",
    8: "Birleştirme: örtüşme ve hata ilişkisi",
    9: "Birleştirme: tam arkta biriken hata",
    10: "Birleştirme: her karede RANSAC ve hibrit yöntem",
    11: "Birleştirme: gürültü duyarlılığı",
    12: "Sanal tarama: kamera görüntüsü, eski ve yeni dedektör",
    13: "Sanal tarama: sapma ve kapsama haritası",
    14: "Sanal tarama: yeniden oluşturulan modelde segmentasyon",
    15: "Sanal tarama: uçtan uca akış",
    16: "Sanal tarama: kalibrasyon, IMU ve gürültü duyarlılığı",
    17: "Sanal tarama: konum senaryoları (ideal, IMU, IMU + ICP)",
    18: "Klinik ölçümler: yapay zekâ ile uzman etiketlerinin karşılaştırması",
}


def sonuclar_bolumu():
    import re
    sekiller = [f for f in os.listdir(GORSEL_DIZIN) if f.endswith(".png")] if os.path.isdir(GORSEL_DIZIN) else []
    if not sekiller:
        return html.P("rapor_gorselleri klasöründe şekil bulunamadı.")

    def no(f):
        m = re.match(r"sekil(\d+)", f)
        return int(m.group(1)) if m else 999
    sekiller.sort(key=no)
    return html.Div([html.Figure([html.A(html.Img(src=f"/gorsel/{s}", alt=SEKIL_ADLARI.get(no(s), s)), href=f"/gorsel/{s}", target="_blank"),
                                  html.Figcaption(f"Şekil {no(s)}. {SEKIL_ADLARI.get(no(s), s)}")]) for s in sekiller],
                    className="galeri")


app.layout = html.Div([
    html.Header([
        html.Div([
            html.H1("DentalScanner"),
            html.P("Lazer çizgili ağız içi tarama ve yapay zekâ ile diş segmentasyonu"),
        ]),
        html.Div([
            html.Span(f"Hesaplama: {'GPU (CUDA)' if cihaz() == 'cuda' else 'İşlemci'}", className="rozet"),
            html.Span("Model: DGCNN, test mIoU 0,817", className="rozet"),
        ], className="rozetler"),
    ], className="ust"),
    dcc.Tabs(id="sekmeler", value="canli", children=[
        dcc.Tab(label="Canlı sanal tarama", value="canli", children=canli),
        dcc.Tab(label="İki çene birlikte", value="kapanis", children=kapanis),
        dcc.Tab(label="Dosyadan segmentasyon", value="yukle", children=yukle),
        dcc.Tab(label="Deney sonuçları", value="sonuc", children=html.Section(sonuclar_bolumu(), className="sonuclar")),
    ], className="sekmeler"),
    dcc.Interval(id="saat", interval=450),
    dcc.Store(id="son-surum", data=-1),
    dcc.Store(id="son-asama", data=None),
    dcc.Store(id="sahne-durum", data=None),
], className="sayfa")


# ───────────────────────── geri cagirmalar ─────────────────────────

@app.callback(Output("gunluk", "children", allow_duplicate=True),
              Input("baslat", "n_clicks"),
              State("vaka", "value"), State("gecisler", "value"), State("adim", "value"),
              State("mod", "value"), State("gurultu", "value"), State("kayit", "value"),
              prevent_initial_call=True)
def baslat(_, vaka, gecisler, adim, mod, gurultu, kayit):
    if not vaka:
        return "Önce bir hasta modeli seçin."
    if not gecisler:
        return "En az bir tarama geçişi seçin."
    sira = [g for g in ("tepe", "yanak", "dil") if g in gecisler]
    if not motor.baslat(vaka, sira, adim, mod, gurultu, kayittan_izin=bool(kayit)):
        return "Bir tarama zaten çalışıyor. Bitmesini bekleyin ya da durdurun."
    return "Tarama başlatılıyor…"


@app.callback(Output("gunluk", "children", allow_duplicate=True), Input("durdur", "n_clicks"), prevent_initial_call=True)
def durdur(_):
    motor.durdur()
    return no_update


@app.callback(
    Output("kamera", "src"), Output("kare-bilgi", "children"), Output("ilerleme-dolu", "style"),
    Output("asamalar", "children"), Output("gunluk", "children"), Output("baslat", "disabled"),
    Output("sahne-durum", "data"), Output("son-surum", "data"), Output("son-asama", "data"),
    Output("odontogram", "children"), Output("metrikler", "children"), Output("gorunum", "value"),
    Output("olcum-panel", "children"),
    Input("saat", "n_intervals"),
    State("gorunum", "value"), State("son-surum", "data"), State("son-asama", "data"),
)
def guncelle(_, gorunum, son_surum, son_asama):
    """Periyodik durum guncellemesi. 3D sahne burada CIZILMEZ: agir sekil bu sik cagrilan geri cagirmada
    uretilirse, bir sonraki periyodik cagri onu gecersiz kilar (Dash ayni geri cagrinin yalnizca son
    istegini uygular) ve gorunum degisikligi kaybolur. Sahne, 'sahne-durum' degisince ayri cizilir."""
    d = motor.oku()
    asama, calisiyor = d["asama"], motor.calisiyor
    src = ("data:image/jpeg;base64," + base64.b64encode(d["goruntu"]).decode()) if d["goruntu"] else no_update
    if d["kare_toplam"]:
        bilgi = f"Kare {d['kare']:,} / {d['kare_toplam']:,}".replace(",", ".") + (f", {d['gecis']} geçişi" if d["gecis"] and asama == "tarama" else "")
        if d["kayittan"]:
            bilgi += " (kayıttan)"
    else:
        bilgi = "Hazır"
    oran = d["kare"] / d["kare_toplam"] if d["kare_toplam"] else 0
    seritte = asama_seridi(asama)
    gunluk = "\n".join(d["mesajlar"][-14:]) + (f"\n\n{d['hata']}" if d["hata"] else "")

    yeni_gorunum = no_update
    if asama == "bitti" and son_asama != "bitti" and d["seg"] is not None:
        yeni_gorunum = gorunum = "seg"
    elif asama in ("hazirlik", "tarama") and son_asama in ("bitti", None) and gorunum != "nokta":
        yeni_gorunum = gorunum = "nokta"           # yeni tarama: canli nokta bulutuna don
    sahne_durum = no_update
    if asama != son_asama or (gorunum == "nokta" and d["nokta_surum"] != son_surum):
        sahne_durum = dict(asama=asama, surum=d["nokta_surum"])

    od, met, olc_panel = no_update, no_update, no_update
    if asama != son_asama:
        r = d["sonuc"]
        if asama == "bitti" and r:
            s = r["segmentasyon"]
            od = odontogram(s["disler"], {int(k): v for k, v in s.get("dis_iou", {}).items()})
            md = r["mesh_dogrulugu_mm"]
            met = [
                metrik(f"{md['rms']:.3f}".replace(".", ","), "mm", "Yüzey sapması (RMS)"),
                metrik(f"%{r['kapsama_dis_0.2mm'] * 100:.1f}".replace(".", ","), "", "Diş yüzeyi kapsaması (0,2 mm)"),
                metrik(f"{s['mIoU_dis']:.3f}".replace(".", ","), "", "Segmentasyon mIoU"),
                metrik(str(len(s["disler"])), "diş", "Bulunan diş"),
                metrik(f"{r['nokta_sayisi'] / 1e3:.0f}", "bin", "3D nokta"),
            ] + ([metrik(f"{r['poz_hatasi_ort_mm']:.2f}".replace(".", ","), "mm", "Ortalama poz hatası")]
                 if r.get("ayar", {}).get("mod") != "ideal" else [])
            o = d["olcum"]
            olc_panel = (olcum_paneli(o["tahmin"], o["gercek"], "Klinik ölçümler: taranan model + yapay zekâ vs. orijinal model + uzman etiketi")
                         if o else html.P("Klinik ölçüm yapılamadı.", className="bos-durum"))
        elif asama in ("hazirlik", "tarama") and son_asama in ("bitti", None, "hazirlik"):
            od, met = odontogram(), html.P("Tarama sürüyor. Bitince ölçümler ve bulunan dişler burada görünür.",
                                            className="bos-durum")
            olc_panel = html.P("Tarama sürüyor. Bitince klinik ölçümler burada görünür.", className="bos-durum")
    return (src, bilgi, {"width": f"{oran * 100:.1f}%"}, seritte, gunluk, calisiyor,
            sahne_durum, d["nokta_surum"], asama, od, met, yeni_gorunum, olc_panel)


@app.callback(Output("sahne", "figure"), Input("gorunum", "value"), Input("sahne-durum", "data"))
def sahne_ciz(gorunum, _):
    d = motor.oku("nokta", "yuzey", "seg", "olcum", "sonuc")
    if gorunum == "nokta":
        return nokta_sekli(d["nokta"])
    if gorunum == "yuzey":
        if d["yuzey"] is None:
            return bos_sekil("Sapma haritası tarama bitince hazır olur")
        vmax = 0.1 if (d["sonuc"] or {}).get("ayar", {}).get("mod") == "ideal" else 0.5
        ust = d["seg"] is not None and _ust_cene_mi(d["seg"]["V"], d["seg"]["etiket"])
        return yuzey_sekli(d["yuzey"], vmax, asagidan=ust)
    if d["seg"] is None:
        return bos_sekil("Segmentasyon tarama bitince hazır olur" if gorunum == "seg" else "Ölçümler tarama bitince hazır olur")
    if gorunum == "olcum":
        return olcum_sekli(d["seg"], (d["olcum"] or {}).get("gorsel"))
    return seg_sekli(d["seg"])


@app.callback(Output("k-tarama", "options"), Output("k-tarama", "value"), Output("k-tarama-kutu", "style"),
              Input("k-hasta", "value"), Input("k-kaynak", "value"), Input("sekmeler", "value"))
def kapanis_tarama_listesi(hasta, kaynak, _):
    if kaynak != "tarama":
        return [], 0, {"display": "none"}
    liste = kapanis_taramalari(hasta) if hasta else []
    return [{"label": s["etiket"], "value": i} for i, s in enumerate(liste)], 0, {}


@app.callback(Output("k-sahne", "figure"), Output("k-odontogram", "children"), Output("k-metrikler", "children"),
              Output("k-bilgi", "children"), Output("k-durum", "children"), Output("k-metrikler", "style"),
              Input("k-hasta", "value"), Input("k-kaynak", "value"), Input("k-tarama", "value"),
              Input("k-renk", "value"), Input("k-aciklik", "value"), Input("sekmeler", "value"))
def kapanis_guncelle(hasta, kaynak, tarama_no, renk, aciklik, sekme):
    if sekme != "kapanis" or not hasta:
        return no_update, no_update, no_update, no_update, no_update, no_update
    if kaynak == "tarama" and not kapanis_taramalari(hasta):
        return (bos_sekil("Bu hastanın iki çenesi henüz aynı ayarlarla taranmadı"), odontogram(), [], "",
                "“Canlı sanal tarama” sekmesinde önce alt, sonra üst çeneyi aynı ayarlarla tarayın "
                "ya da “Veri setindeki orijinal” seçeneğine geçin.", no_update)
    try:
        alt, ust, R, t, bilgi = _kapanis_verisi(hasta, kaynak, tarama_no)
    except Exception as e:
        return bos_sekil("Model hazırlanamadı"), no_update, no_update, "", f"Hata: {e}", no_update
    aciklik = float(aciklik or 0)
    if ctx.triggered_id == "k-aciklik":            # sadece ust cene kayar: tum sekli yeniden gondermeye gerek yok
        p = Patch()
        p["data"][1]["z"] = (ust["V"] @ R.T + t)[:, 2] + aciklik
        return p, no_update, no_update, no_update, no_update, no_update
    if renk == "sapma" and alt["sapma"] is None:
        renk = "seg"
    vmax = 0.1 if "konum biliniyor" in bilgi else 0.5
    fig = kapanis_sekli(alt, ust, R, t, aciklik, renk, vmax)
    od = odontogram(sorted(set(alt["disler"]) | set(ust["disler"])), {**alt["dis_iou"], **ust["dis_iou"]})
    ma, mu = _cene_metrikleri("Alt çene", alt), _cene_metrikleri("Üst çene", ust)
    met = ma + mu + _bolton_kartlari(alt, ust)
    durum = "" if renk == "seg" or alt["sapma"] is not None else "Sapma haritası yalnızca taranan modellerde var."
    sutun = {"gridTemplateColumns": f"repeat({len(ma)}, minmax(0, 1fr))"}
    return fig, od, met, f"{hasta} · {bilgi}", durum, sutun


@app.callback(Output("yukleme-durum", "children"), Output("yukleme-sahne", "figure"),
              Output("yukleme-odontogram", "children"), Output("yukleme-olcum", "children"),
              Input("yukleme", "contents"), State("yukleme", "filename"), prevent_initial_call=True)
def yuklenen(icerik, ad):
    if not icerik:
        return no_update, no_update, no_update, no_update
    uzanti = os.path.splitext(ad or "")[1].lower()
    if uzanti not in (".obj", ".ply", ".stl"):
        return f"{ad}: desteklenmeyen dosya türü. .obj, .ply veya .stl yükleyin.", no_update, no_update, no_update
    veri = base64.b64decode(icerik.split(",", 1)[1])
    with tempfile.NamedTemporaryFile(suffix=uzanti, delete=False) as fh:
        fh.write(veri)
        yol = fh.name
    try:
        gos, seg = motor.dosya_segmentle(yol)
    except Exception as e:
        return f"{ad} işlenemedi: {e}", no_update, no_update, no_update
    finally:
        try:
            os.remove(yol)
        except OSError:
            pass
    durum = (f"{ad}: {len(seg['disler'])} diş bulundu ({seg['cihaz'].upper() if seg['cihaz'] == 'cuda' else 'işlemci'}, "
             f"{seg['sure']:.0f} sn).")
    o = seg.get("olcum")
    return (durum, olcum_sekli(gos, o["gorsel"]) if o else seg_sekli(gos), odontogram(seg["disler"]),
            olcum_paneli(o["ozet"]) if o else None)


if __name__ == "__main__":
    port = int(os.environ.get("DEMO_PORT", 8050))
    if not os.environ.get("DEMO_TARAYICI_ACMA"):
        threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    print(f"\n  DentalScanner demo: http://127.0.0.1:{port}   (kapatmak için bu pencerede Ctrl+C)\n")
    app.run(host="127.0.0.1", port=port, debug=False)
