"""
Segmentasyondan otomatik ortodontik olcumler.

Girdi: cene modeli (kose noktalari, istege bagli ucgenler) + her kose icin FDI dis numarasi (0 = dis eti).
Cikti (hepsi mm):
  * Her disin meziodistal genisligi: dis kuronunun, ark boyunca (komsu dislerin merkezlerinden gecen
    dogrultu) okluzal duzlemdeki uzunlugu.
  * Mevcut ark uzunlugu (available space): dislerin okluzal merkezlerinden gecen duzgun ark egrisinin
    (4. derece polinom) iki taraftaki birinci buyuk azi disinin meziyal kenari arasindaki uzunlugu.
  * Gerekli alan (required space): iki taraftaki 1-5 numarali dislerin (10 dis) genislikleri toplami.
  * Ark uzunlugu farki = gerekli - mevcut. Pozitifse capraskilik (dislerin sigmadigi alan), negatifse
    dis araliklari (diastema) vardir.
  * Interkanin genislik: iki kopek disinin tepe noktalari arasi; intermolar genislik: iki birinci buyuk
    azinin okluzal merkezleri arasi.
  * Bolton analizi (iki cene birlikte): on (3-3) ve toplam (6-6) oranlari.

Not: Klinik standart, ortodontistin kumpasla ya da dijital modelde elle isaretledigi temas noktalaridir.
Burada ayni tanimlar dis etiketlerinden otomatik cikarilir. Dogrulama icin ayni algoritma uzmanin
etiketledigi gercek segmentasyona uygulanir; aradaki fark segmentasyon hatasinin olcume etkisidir.
"""
import numpy as np

SURUM = 3                       # algoritma degisirse artirilir (kayitli olcumler yeniden hesaplanir)
YUZDELIK = 2.0                  # genislik icin uc noktalar: kuron noktalarinin bu yuzdelik dilimleri
UC_TEGET = "komsu"              # ark sonundaki disler icin dogrultu: 'egri' ya da 'komsu'
ON_DISLER = (1, 2, 3)           # kesiciler + kopek
GEREKLI_ALAN = (1, 2, 3, 4, 5)  # 5-5 arasi
BOLTON_ON_IDEAL = (77.2, 1.65)
BOLTON_TOPLAM_IDEAL = (91.3, 1.91)


def cene_turu(fdi):
    q = np.asarray(fdi)[np.asarray(fdi) > 0] // 10
    if len(q) == 0:
        return None
    return "ust" if np.isin(q, (1, 2)).mean() >= 0.5 else "alt"


def ark_sirasi(cene):
    """Hastanin sagindan soluna FDI sirasi."""
    sag, sol = (1, 2) if cene == "ust" else (4, 3)
    return [sag * 10 + k for k in range(8, 0, -1)] + [sol * 10 + k for k in range(1, 9)]


def _sag_mi(f):
    return f // 10 in (1, 4)


# ───────────────────────── etiket temizligi ─────────────────────────

def _ucgen_alanlari(V, F):
    return 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)


def _bilesenler(F, n):
    """Ucgen baglantisina gore kose bilesenleri (birlesim-bul, vektorel yol sikistirma)."""
    ebeveyn = np.arange(n)
    kenar = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [0, 2]]])
    while True:
        a, b = ebeveyn[kenar[:, 0]], ebeveyn[kenar[:, 1]]
        fark = a != b
        if not fark.any():
            break
        kucuk, buyuk = np.minimum(a[fark], b[fark]), np.maximum(a[fark], b[fark])
        np.minimum.at(ebeveyn, buyuk, kucuk)
        while True:                                   # yol sikistirma
            yeni = ebeveyn[ebeveyn]
            if np.array_equal(yeni, ebeveyn):
                break
            ebeveyn = yeni
    return ebeveyn


def dis_maskeleri(V, fdi, F=None, min_alan_mm2=12.0, min_nokta=40):
    """{fdi: kose maskesi}. Her etiketin yalnizca en buyuk bagli parcasi alinir (sacilmis yanlis etiketler
    atilir); cok kucuk parcalar 'dis yok' sayilir."""
    V, fdi = np.asarray(V, float), np.asarray(fdi)
    sonuc = {}
    for f in np.unique(fdi):
        if f <= 0:
            continue
        m = fdi == f
        if m.sum() < min_nokta:
            continue
        if F is not None and len(F):
            Fm = F[m[F].all(1)]
            if len(Fm) == 0:
                continue
            kok = _bilesenler(Fm, len(V))
            alan = _ucgen_alanlari(V, Fm)
            kokler, ters = np.unique(kok[Fm[:, 0]], return_inverse=True)
            toplam = np.bincount(ters, alan)
            en = kokler[np.argmax(toplam)]
            if toplam.max() < min_alan_mm2:
                continue
            mm = np.zeros(len(V), bool)
            mm[np.unique(Fm[kok[Fm[:, 0]] == en])] = True
        else:                                          # nokta bulutu: medyana yakin noktalar
            P = V[m]
            d = np.linalg.norm(P - np.median(P, 0), axis=1)
            mm = np.zeros(len(V), bool)
            mm[np.where(m)[0][d < max(6.0, 2.5 * np.median(d))]] = True
        if mm.sum() >= min_nokta:
            sonuc[int(f)] = mm
    return sonuc


# ───────────────────────── geometri ─────────────────────────

def okluzal_cerceve(V, fdi, maskeler):
    """Okluzal duzlem cercevesi: O (orijin), R (satirlar: x = hastanin sagindan soluna, y = arkadan one,
    n = dis tepelerine dogru)."""
    V = np.asarray(V, float)
    merkez = {f: V[m].mean(0) for f, m in maskeler.items()}
    C = np.array(list(merkez.values()))
    O = C.mean(0)
    if len(C) >= 3:
        _, _, Vt = np.linalg.svd(C - O)
        n = Vt[2]
    else:
        n = np.array([0, 0, 1.0])
    dis = np.any(np.array(list(maskeler.values())), 0) if maskeler else np.zeros(len(V), bool)
    if dis.any() and (~dis).any():
        if np.dot(V[dis].mean(0) - V[~dis].mean(0), n) < 0:
            n = -n
    elif n[2] < 0:
        n = -n
    sag = [c for f, c in merkez.items() if _sag_mi(f)]
    sol = [c for f, c in merkez.items() if not _sag_mi(f)]
    if sag and sol:
        x = np.mean(sol, 0) - np.mean(sag, 0)
    else:
        x = np.cross([0, 1.0, 0], n)
    x = x - np.dot(x, n) * n
    x /= np.linalg.norm(x) + 1e-12
    y = np.cross(n, x)
    on = [c for f, c in merkez.items() if f % 10 <= 2]
    arka = [c for f, c in merkez.items() if f % 10 >= 5]
    if on and arka and np.dot(np.mean(on, 0) - np.mean(arka, 0), y) < 0:
        x, y = -x, -y              # sag/sol kumesi yoksa yonu on dislerle duzelt
    return O, np.array([x, y, n])


def olc(V, fdi, F=None, cene=None):
    """Tek cene icin tum olcumler. V: (N,3) mm, fdi: (N,) FDI numarasi (0 = dis eti)."""
    V = np.asarray(V, float)
    fdi = np.asarray(fdi).astype(int)
    F = None if F is None else np.asarray(F, int)
    cene = cene or cene_turu(fdi)
    if cene is None:
        return dict(cene=None, disler={}, ark=None, interkanin=None, intermolar=None, uyari="Diş bulunamadı.")
    maskeler = {f: m for f, m in dis_maskeleri(V, fdi, F).items() if (f // 10 in (1, 2)) == (cene == "ust")}
    if len(maskeler) < 2:
        return dict(cene=cene, disler={}, ark=None, interkanin=None, intermolar=None, uyari="Yeterli diş yok.")
    O, R = okluzal_cerceve(V, fdi, maskeler)
    P = (V - O) @ R.T                                  # x, y, yukseklik
    sira = [f for f in ark_sirasi(cene) if f in maskeler]
    merkez2 = {f: P[maskeler[f], :2].mean(0) for f in sira}

    # Ark dogrultusu: dis merkezlerinden gecen duzgun parametrik egri (x(s), y(s)); s = kumulatif kiris
    # uzunlugu. Yerinden oynamis tek bir dis, komsu merkezlerden hesaplanan dogrultuyu bozar; duzgun egri bozmaz.
    C = np.array([merkez2[f] for f in sira])
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
    derece = min(4, len(sira) - 1) if len(sira) >= 5 else 1
    kx, ky = np.polyfit(s, C[:, 0], derece), np.polyfit(s, C[:, 1], derece)
    teget = {f: np.array([np.polyval(np.polyder(kx), si), np.polyval(np.polyder(ky), si)]) for f, si in zip(sira, s)}

    disler = {}
    for i, f in enumerate(sira):
        t = teget[f]
        if UC_TEGET == "komsu" and len(sira) >= 3 and i in (0, len(sira) - 1):
            t = merkez2[sira[1]] - merkez2[sira[0]] if i == 0 else merkez2[sira[-1]] - merkez2[sira[-2]]
        t = t / (np.linalg.norm(t) + 1e-12)
        Q = P[maskeler[f]]
        s = Q[:, :2] @ t
        a, b = np.percentile(s, YUZDELIK), np.percentile(s, 100 - YUZDELIK)
        # Uclardan kirpma (sinirdaki yanlis etiketlere karsi) genisligi kisaltir. Kuron yuzeyindeki noktalar
        # ark yonunde yaklasik duzgun dagildigindan kirpilan pay geri eklenir (veride dogrulandi: kirpilmis /
        # kirpilmamis oran 0,956, duzgun dagilim beklentisi 0,970).
        pay = (b - a) * (YUZDELIK / 100) / (1 - 2 * YUZDELIK / 100)
        a, b = a - pay, b + pay
        h = Q[:, 2]
        ust = Q[h >= np.percentile(h, 60)]               # kuronun okluzal kismi
        orta = ust[:, :2].mean(0)
        dik = np.array([-t[1], t[0]])
        o = np.dot(orta, dik)
        hz = float(np.percentile(h, 85))
        uc_sag = np.r_[a * t + o * dik, hz]               # ark sirasinda once gelen uc (hastanin sagina dogru)
        uc_sol = np.r_[b * t + o * dik, hz]
        tepe = Q[np.argmax(h)]
        disler[f] = dict(genislik=float(b - a), merkez2=orta, uc_sag=uc_sag, uc_sol=uc_sol, tepe=tepe,
                         okluzal_merkez=ust.mean(0))

    def mezial(f):
        if f not in disler:
            return None
        return disler[f]["uc_sol"] if _sag_mi(f) else disler[f]["uc_sag"]

    def distal(f):
        if f not in disler:
            return None
        return disler[f]["uc_sag"] if _sag_mi(f) else disler[f]["uc_sol"]

    q_sag, q_sol = (1, 2) if cene == "ust" else (4, 3)
    ark = None
    uyari = []
    sol_uc = mezial(q_sol * 10 + 6) if (q_sol * 10 + 6) in disler else distal(q_sol * 10 + 5)
    sag_uc = mezial(q_sag * 10 + 6) if (q_sag * 10 + 6) in disler else distal(q_sag * 10 + 5)
    ark_disleri = [f for f in sira if f % 10 <= 6]
    if sol_uc is not None and sag_uc is not None and len(ark_disleri) >= 5:
        # Parametrik ark egrisi (x(s), y(s)): U seklindeki dar arklarda y(x) bicimi yan kisimlarda dikeylesip
        # patlar; kiris uzunlugu parametresi her ark biciminde kararlidir.
        X = np.array([disler[f]["merkez2"] for f in ark_disleri])
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))]
        derece = min(4, len(X) - 1) if len(X) >= 7 else 2
        kx, ky = np.polyfit(s, X[:, 0], derece), np.polyfit(s, X[:, 1], derece)
        ss = np.linspace(-0.15 * s[-1], 1.15 * s[-1], 3000)
        egri2 = np.c_[np.polyval(kx, ss), np.polyval(ky, ss)]

        def izdusum(p):
            return ss[np.argmin(np.linalg.norm(egri2 - p[:2], axis=1))]
        s0, s1 = sorted([izdusum(sag_uc), izdusum(sol_uc)])
        aralik = (ss >= s0) & (ss <= s1)
        xs, ys = egri2[aralik, 0], egri2[aralik, 1]
        mevcut = float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))
        gerekli_disler = [q * 10 + k for q in (q_sag, q_sol) for k in GEREKLI_ALAN]
        eksik = [f for f in gerekli_disler if f not in disler]
        gerekli = float(sum(disler[f]["genislik"] for f in gerekli_disler if f in disler))
        hz = np.mean([disler[f]["uc_sag"][2] for f in ark_disleri])
        egri = O + np.c_[xs, ys, np.full_like(xs, hz)] @ R
        ark = dict(mevcut=mevcut, gerekli=gerekli if not eksik else None,
                   fark=(gerekli - mevcut) if not eksik else None, eksik=eksik, egri=egri)
        if eksik:
            uyari.append("Gerekli alan için eksik diş: " + ", ".join(map(str, eksik)))
    else:
        uyari.append("Ark uzunluğu için iki taraftaki 6 (ya da 5) numaralı diş gerekli.")

    def mesafe(f1, f2, alan):
        if f1 in disler and f2 in disler:
            return float(np.linalg.norm(disler[f1][alan][:2] - disler[f2][alan][:2]))
        return None

    interkanin = mesafe(q_sag * 10 + 3, q_sol * 10 + 3, "tepe")
    intermolar = mesafe(q_sag * 10 + 6, q_sol * 10 + 6, "okluzal_merkez")

    # dunya koordinatlarina (gorsellestirme icin)
    def dunya(p):
        return O + np.asarray(p) @ R
    for f, d in disler.items():
        d["uc_sag_3d"] = dunya(d["uc_sag"])
        d["uc_sol_3d"] = dunya(d["uc_sol"])
    return dict(cene=cene, disler=disler, ark=ark, interkanin=interkanin, intermolar=intermolar,
                uyari=" ".join(uyari) if uyari else None, cerceve=(O, R))


def bolton(alt, ust):
    """Iki cenenin olcumlerinden Bolton oranlari (yuzde). Eksik dis varsa ilgili oran None."""
    def toplam(o, q1, q2, kler):
        fl = [q * 10 + k for q in (q1, q2) for k in kler]
        if any(f not in o["disler"] for f in fl):
            return None
        return sum(o["disler"][f]["genislik"] for f in fl)
    sonuc = {}
    for ad, kler in (("on", ON_DISLER), ("toplam", (1, 2, 3, 4, 5, 6))):
        a, u = toplam(alt, 4, 3, kler), toplam(ust, 1, 2, kler)
        sonuc[ad] = (100.0 * a / u) if (a and u) else None
    return sonuc


def ozet(o):
    """JSON'a yazilabilir kisa ozet."""
    return dict(cene=o["cene"],
                genislik={int(f): round(d["genislik"], 3) for f, d in o["disler"].items()},
                ark_mevcut=None if not o["ark"] else round(o["ark"]["mevcut"], 3),
                ark_gerekli=None if not o["ark"] or o["ark"]["gerekli"] is None else round(o["ark"]["gerekli"], 3),
                ark_farki=None if not o["ark"] or o["ark"]["fark"] is None else round(o["ark"]["fark"], 3),
                interkanin=None if o["interkanin"] is None else round(o["interkanin"], 3),
                intermolar=None if o["intermolar"] is None else round(o["intermolar"], 3),
                uyari=o.get("uyari"))
