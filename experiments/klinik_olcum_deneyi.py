"""
Klinik olcum dogrulama deneyi.

Test setindeki taramalarda ayni olcum algoritmasi iki kez calistirilir:
  (1) uzmanin etiketledigi gercek segmentasyon  -> referans
  (2) DGCNN'in tahmin ettigi segmentasyon        -> otomatik olcum
Fark, segmentasyon hatasinin klinik olcume etkisini gosterir. Ayrica sanal tarayicidan elde edilen
yeniden olusturulmus modeller (runs/demo kayitlari) uzerindeki olcumler orijinal model + gercek
etiketlerle karsilastirilir (tarama + segmentasyon hatasinin toplam etkisi).

Kullanim:
  python experiments/klinik_olcum_deneyi.py --veri data/3DTeethData --liste secim.json --out runs/klinik_olcum
  python experiments/klinik_olcum_deneyi.py --sadece-ciz runs/klinik_olcum/sonuclar.json --figs rapor_gorselleri
"""
import os
import sys
import json
import glob
import time
import argparse
import numpy as np

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from ai_model.dataset import obj_oku, mesh_oku, kose_normalleri, FDI_SINIFLARI
from processing.klinik_olcum import olc, ozet

FDI = np.array(FDI_SINIFLARI)
TUR_ADI = {1: "Orta kesici", 2: "Yan kesici", 3: "Köpek", 4: "1. küçük azı", 5: "2. küçük azı",
           6: "1. büyük azı", 7: "2. büyük azı", 8: "3. büyük azı"}


def tahmin_et(model, kayit, v, f, cihaz="cpu", tur=1):
    from ai_model.inference import koseleri_tahmin_et
    n = kose_normalleri(v, f)
    t, _ = koseleri_tahmin_et(model, v.astype(np.float32), n, nokta_sayisi=int(kayit.get("num_points", 8192)),
                              cihaz=cihaz, tur_sayisi=tur, batch_boyutu=8 if cihaz == "cuda" else 1)
    return t


def karsilastir(gercek, tahmin):
    """Iki ozet arasindaki farklar (tahmin - gercek)."""
    ortak = sorted(set(gercek["genislik"]) & set(tahmin["genislik"]))
    d = dict(genislik=[(f, gercek["genislik"][f], tahmin["genislik"][f]) for f in ortak],
             sadece_gercek=sorted(set(gercek["genislik"]) - set(tahmin["genislik"])),
             sadece_tahmin=sorted(set(tahmin["genislik"]) - set(gercek["genislik"])))
    for k in ("ark_mevcut", "ark_gerekli", "ark_farki", "interkanin", "intermolar"):
        d[k] = (gercek[k], tahmin[k])
    return d


def test_seti(veri, liste, cikti, cihaz, tur):
    import torch
    from ai_model.network import checkpoint_yukle
    model, kayit = checkpoint_yukle(os.path.join(KOK, "ai_model", "best_dental_model.pth"), cihaz)
    sonuc = []
    for i, (o, j) in enumerate(liste):
        t0 = time.time()
        yol = os.path.join(veri, o)
        v, f = obj_oku(yol)
        _, _, etiket, _ = mesh_oku(yol, os.path.join(veri, j))
        g = olc(v, FDI[etiket], f)
        onbellek = os.path.join(cikti, "tahminler", os.path.splitext(os.path.basename(o))[0] + ".npy")
        if os.path.exists(onbellek):                    # olcum algoritmasi degisirse yeniden cikarim gerekmez
            p = np.load(onbellek)
        else:
            with torch.no_grad():
                p = tahmin_et(model, kayit, v, f, cihaz, tur)
            os.makedirs(os.path.dirname(onbellek), exist_ok=True)
            np.save(onbellek, p.astype(np.int8))
        t = olc(v, FDI[p], f, cene=g["cene"])
        gs, ts = ozet(g), ozet(t)
        sonuc.append(dict(tarama=os.path.basename(o), gercek=gs, tahmin=ts, fark=karsilastir(gs, ts)))
        print(f"[{i + 1}/{len(liste)}] {os.path.basename(o)}  {time.time() - t0:.0f} sn  "
              f"ark farki gercek {gs['ark_farki']} / tahmin {ts['ark_farki']}", flush=True)
        with open(os.path.join(cikti, "sonuclar.json"), "w", encoding="utf-8") as fh:
            json.dump(sonuc, fh, indent=1, ensure_ascii=False)
    return sonuc


def sanal_taramalar(veri):
    """runs/demo kayitlarindaki yeniden olusturulmus modeller: tahmin edilen etiketlerle olcum,
    orijinal model + gercek etiketlerle karsilastirma."""
    sonuc = []
    for d in sorted(glob.glob(os.path.join(KOK, "runs", "demo", "*", "sonuc.json"))):
        r = json.load(open(d, encoding="utf-8"))
        ayar = r["ayar"]
        ad = os.path.splitext(ayar["vaka"])[0]
        bulunan = glob.glob(os.path.join(veri, "*", "*", ad.split("_")[0], ad + ".obj"))
        if not bulunan:
            continue
        v, f = obj_oku(bulunan[0])
        _, _, etiket, _ = mesh_oku(bulunan[0], os.path.splitext(bulunan[0])[0] + ".json")
        g = ozet(olc(v, FDI[etiket], f))
        gz = np.load(os.path.join(os.path.dirname(d), "gosterim.npz"))
        t = ozet(olc(gz["V"], FDI[gz["etiket"].astype(int)], gz["F"], cene=g["cene"]))
        sonuc.append(dict(tarama=ad, ayar=ayar, gercek=g, tahmin=t, fark=karsilastir(g, t)))
    return sonuc


def istatistik(sonuc):
    G = [(f, a, b) for s in sonuc for f, a, b in s["fark"]["genislik"]]
    fark = np.array([b - a for _, a, b in G]) if G else np.zeros(0)
    ist = dict(tarama=len(sonuc), dis=len(G),
               genislik_mae=float(np.abs(fark).mean()) if len(G) else None,
               genislik_medyan=float(np.median(np.abs(fark))) if len(G) else None,
               genislik_yanlilik=float(fark.mean()) if len(G) else None,
               genislik_sd=float(fark.std(ddof=1)) if len(G) > 1 else None,
               genislik_05mm=float((np.abs(fark) <= 0.5).mean()) if len(G) else None,
               dis_turu={})
    for k, ad in TUR_ADI.items():
        x = np.array([b - a for f, a, b in G if f % 10 == k])
        if len(x):
            ist["dis_turu"][ad] = dict(n=int(len(x)), mae=float(np.abs(x).mean()))
    for k in ("ark_mevcut", "ark_gerekli", "ark_farki", "interkanin", "intermolar"):
        x = np.array([s["fark"][k][1] - s["fark"][k][0] for s in sonuc
                      if s["fark"][k][0] is not None and s["fark"][k][1] is not None])
        ist[k] = dict(n=int(len(x)), mae=float(np.abs(x).mean()) if len(x) else None,
                      yanlilik=float(x.mean()) if len(x) else None,
                      maks=float(np.abs(x).max()) if len(x) else None)
    # capraskilik siniflamasi: <=0 yok, 0-4 hafif, 4-8 orta, >8 siddetli
    def sinif(x):
        return 0 if x <= 0 else 1 if x <= 4 else 2 if x <= 8 else 3
    c = [(sinif(s["fark"]["ark_farki"][0]), sinif(s["fark"]["ark_farki"][1])) for s in sonuc
         if s["fark"]["ark_farki"][0] is not None and s["fark"]["ark_farki"][1] is not None]
    ist["capraskilik_sinifi_uyum"] = dict(n=len(c), uyum=float(np.mean([a == b for a, b in c])) if c else None)
    ist["eksik_bulunan"] = int(sum(len(s["fark"]["sadece_gercek"]) for s in sonuc))
    ist["fazla_bulunan"] = int(sum(len(s["fark"]["sadece_tahmin"]) for s in sonuc))
    return ist


def ciz(sonuc, figs, sanal=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    G = [(f, a, b) for s in sonuc for f, a, b in s["fark"]["genislik"]]
    a = np.array([x[1] for x in G]); b = np.array([x[2] for x in G]); tur = np.array([x[0] % 10 for x in G])
    renk = {1: "#1f4e79", 2: "#2f6ea5", 3: "#7fa7cf", 4: "#c0392b", 5: "#e39b3b", 6: "#2e7d32", 7: "#66bb6a", 8: "#9e9e9e"}
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.6))

    ax = axs[0]
    for k in sorted(set(tur)):
        m = tur == k
        ax.scatter(a[m], b[m], s=10, color=renk[k], label=TUR_ADI[k], alpha=0.8, linewidths=0)
    lo, hi = min(a.min(), b.min()) - 0.5, max(a.max(), b.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], color="#555", lw=1, ls="--")
    ax.set_xlabel("Gerçek etiketlerden genişlik (mm)"); ax.set_ylabel("Yapay zekâ segmentasyonundan (mm)")
    ax.set_title("Diş genişliği (meziodistal)"); ax.legend(frameon=False, fontsize=7.5, markerscale=2)

    ax = axs[1]
    ort, fark = (a + b) / 2, b - a
    for k in sorted(set(tur)):
        m = tur == k
        ax.scatter(ort[m], fark[m], s=10, color=renk[k], alpha=0.8, linewidths=0)
    md, sd = fark.mean(), fark.std(ddof=1)
    for y, ls in [(md, "-"), (md + 1.96 * sd, "--"), (md - 1.96 * sd, "--")]:
        ax.axhline(y, color="#555", lw=1, ls=ls)
    ax.text(ax.get_xlim()[1] if False else ort.max(), md + 1.96 * sd, f" +1,96 SS: {md + 1.96 * sd:.2f}".replace(".", ","), va="bottom", ha="right", fontsize=8)
    ax.text(ort.max(), md - 1.96 * sd, f" −1,96 SS: {md - 1.96 * sd:.2f}".replace(".", ","), va="top", ha="right", fontsize=8)
    ax.set_xlabel("İki ölçümün ortalaması (mm)"); ax.set_ylabel("Fark: yapay zekâ − gerçek (mm)")
    ax.set_title(f"Bland–Altman (ortalama fark {md:+.2f} mm)".replace(".", ","))

    ax = axs[2]
    X = [(s["fark"]["ark_farki"][0], s["fark"]["ark_farki"][1]) for s in sonuc if None not in s["fark"]["ark_farki"]]
    if X:
        X = np.array(X)
        ax.scatter(X[:, 0], X[:, 1], s=26, color="#1f4e79", label=f"Test seti ({len(X)} tarama)", zorder=3)
    if sanal:
        S = [(s["fark"]["ark_farki"][0], s["fark"]["ark_farki"][1], s["ayar"]["mod"]) for s in sanal
             if None not in s["fark"]["ark_farki"]]
        isim = {"ideal": "Sanal tarama (konum biliniyor)", "imu": "Sanal tarama (IMU + ICP)", "imu_ham": "Sanal tarama (sadece IMU)"}
        for mod, m in (("ideal", "D"), ("imu", "s"), ("imu_ham", "x")):
            p = [(x, y) for x, y, mm in S if mm == mod]
            if p:
                p = np.array(p)
                ax.scatter(p[:, 0], p[:, 1], s=40, marker=m, color="#c0392b", label=isim[mod], zorder=4)
    tum = np.concatenate([X.ravel() if len(X) else np.zeros(0)] + ([np.array([v for s in S for v in s[:2]])] if sanal and S else []))
    lo, hi = tum.min() - 1, tum.max() + 1
    ax.plot([lo, hi], [lo, hi], color="#555", lw=1, ls="--")
    ax.axhline(0, color="#ccc", lw=0.8); ax.axvline(0, color="#ccc", lw=0.8)
    ax.set_xlabel("Gerçek etiketlerden (mm)"); ax.set_ylabel("Otomatik ölçüm (mm)")
    ax.set_title("Ark uzunluğu farkı (+ çapraşıklık, − boşluk)"); ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    yol = os.path.join(figs, "sekil18_klinik_olcum.png")
    fig.savefig(yol, dpi=200); plt.close(fig)
    return yol


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--veri", default=os.path.join(KOK, "data", "3DTeethData"))
    p.add_argument("--liste", help="[[obj, json], ...] (veri klasorune gore); verilmezse split.json test seti")
    p.add_argument("--out", default=os.path.join(KOK, "runs", "klinik_olcum"))
    p.add_argument("--figs", default=os.path.join(KOK, "rapor_gorselleri"))
    p.add_argument("--tur", type=int, default=1)
    p.add_argument("--sadece-ciz", action="store_true")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.sadece_ciz:
        sonuc = json.load(open(os.path.join(a.out, "sonuclar.json"), encoding="utf-8"))
    else:
        import torch
        cihaz = "cuda" if torch.cuda.is_available() else "cpu"
        liste = json.load(open(a.liste)) if a.liste else json.load(open(os.path.join(KOK, "runs", "dgcnn", "split.json")))["test"]
        sonuc = test_seti(a.veri, liste, a.out, cihaz, a.tur)
    sanal = sanal_taramalar(a.veri)
    ist = dict(test_seti=istatistik(sonuc), sanal_tarama=[
        dict(tarama=s["tarama"], ayar=s["ayar"], istatistik=istatistik([s])) for s in sanal])
    with open(os.path.join(a.out, "sanal_tarama.json"), "w", encoding="utf-8") as fh:
        json.dump(sanal, fh, indent=1, ensure_ascii=False)
    with open(os.path.join(a.out, "istatistik.json"), "w", encoding="utf-8") as fh:
        json.dump(ist, fh, indent=1, ensure_ascii=False)
    print(json.dumps(ist["test_seti"], indent=1, ensure_ascii=False))
    print(ciz(sonuc, a.figs, sanal))


if __name__ == "__main__":
    main()
