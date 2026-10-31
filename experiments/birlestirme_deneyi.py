"""
Parca Tarama Birlestirme (Stitching) Deneyi — data/four/partial_scans

Fikir
-----
data/four icindeki her dosya, tam alt/ust cenenin bir disi merkez alan KIRPILMIS parcasidir ve
orijinal cene koordinatinda durur. Bu yuzden her parcanin dogru konumu (ground truth) bilinir.

Algoritmaya bu koordinatlar VERILMEZ:
  1. Her parca yuzeyinden bagimsiz olarak yeniden orneklenir (komsu parcalarda ayni noktalar kalmaz).
  2. Sensor gurultusu (Gauss, sigma mm) eklenir.
  3. Parca rastgele dondurulup otelenerek "konumu bilinmeyen tarama" haline getirilir (T_pert).
Birlestirici (processing/icp_stitcher.py) sadece sekle bakarak T_est'i bulur.
Degerlendirme: E = T_est * T_pert birim matris olmalidir. Hata, parcanin noktalarinin
gercek konumundan ne kadar uzaga yerlestirildigi (mm, RMSE) ve donme hatasi (derece) ile olculur.

Deneyler
--------
A  Parca cifti: 3 yontem (ICP nokta-nokta, ICP nokta-duzlem, RANSAC+ICP) x 3 hareket seviyesi
B  Tam ark zincirleme birlestirme: kare->model (mevcut hat) ve kare->kare; biriken kayma (drift)
C  Gurultu duyarliligi: sigma = 0 ... 0.2 mm

Kullanim:
  python experiments/birlestirme_deneyi.py --data data/four/partial_scans --out runs/birlestirme
"""
import os
import re
import sys
import json
import time
import glob
import argparse
import zlib
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from ai_model.dataset import obj_oku
from processing.icp_stitcher import PointCloudStitcher
from utils.logger import get_logger

kayitci = get_logger("birlestirme_deneyi")

HAREKET = {  # (aci_min, aci_max derece, oteleme_min, oteleme_max mm)
    "kucuk": (2.0, 5.0, 1.0, 3.0),     # ardisik iki kare arasi gercekci prob hareketi
    "orta": (10.0, 15.0, 4.0, 6.0),
    "buyuk": (25.0, 35.0, 8.0, 12.0),  # takip kaybi / yeniden konumlama
}
YONTEMLER = {
    "ICP (nokta-nokta)": dict(icp_method="point_to_point", registration_mode="local"),
    "ICP (nokta-düzlem)": dict(icp_method="point_to_plane", registration_mode="local"),
    "RANSAC + ICP (nokta-düzlem)": dict(icp_method="point_to_plane", registration_mode="global"),
    "Hibrit (ICP → gerekirse RANSAC)": dict(icp_method="point_to_plane", registration_mode="hybrid"),
}
BASARI_ESIGI_MM = 0.1


# ───────────────────────── yardimcilar ─────────────────────────

def sabit_hash(*parcalar):
    """Calistirmalar arasi tekrarlanabilir hash (Python hash() her surecte degisir)."""
    return zlib.crc32("|".join(map(str, parcalar)).encode())


def ark_sirasi(dis_no, cene):
    """Parcalari ark boyunca sirala: alt cene 48..41 -> 31..38, ust cene 18..11 -> 21..28."""
    q, i = divmod(dis_no, 10)
    sag, sol = (4, 3) if cene == "lower" else (1, 2)
    return -i if q == sag else i


def hasta_parcalari(kok):
    """{hasta: [(dis_no, obj_yolu), ...]} ark sirasinda."""
    sonuc = {}
    for d in sorted(glob.glob(os.path.join(kok, "*"))):
        dosyalar = []
        for f in glob.glob(os.path.join(d, "*.obj")):
            m = re.search(r"_(lower|upper)_(\d{2})_", os.path.basename(f))
            if m:
                dosyalar.append((int(m.group(2)), m.group(1), f))
        if len(dosyalar) >= 3:
            cene = dosyalar[0][1]
            dosyalar.sort(key=lambda t: ark_sirasi(t[0], cene))
            sonuc[os.path.basename(d)] = [(n, f) for n, _, f in dosyalar]
    return sonuc


_mesh_onbellek = {}


def parca_mesh(yol):
    if yol not in _mesh_onbellek:
        v, f = obj_oku(yol)
        m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v.astype(np.float64)),
                                      o3d.utility.Vector3iVector(f.astype(np.int32)))
        _mesh_onbellek[yol] = (m, v)
    return _mesh_onbellek[yol]


def tarama_simule_et(yol, tohum, sigma, yogunluk=4.0, maks_nokta=20000):
    """Parcayi bagimsiz bir tarama gibi ornekler: yuzeyden yeniden ornekleme + gurultu.

    Returns:
        temiz (N,3) gercek konumdaki noktalar (degerlendirme icin), gurultulu (N,3)
    """
    m, _ = parca_mesh(yol)
    n = int(min(maks_nokta, max(3000, m.get_surface_area() * yogunluk)))
    o3d.utility.random.seed(int(tohum))
    temiz = np.asarray(m.sample_points_uniformly(n).points)
    rng = np.random.RandomState(tohum)
    return temiz, temiz + rng.normal(0.0, sigma, temiz.shape)


def rastgele_hareket(rng, seviye, merkez):
    a0, a1, t0, t1 = HAREKET[seviye]
    eksen = rng.normal(size=3); eksen /= np.linalg.norm(eksen)
    R = Rotation.from_rotvec(np.radians(rng.uniform(a0, a1)) * eksen).as_matrix()
    yon = rng.normal(size=3); yon /= np.linalg.norm(yon)
    t = yon * rng.uniform(t0, t1)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = merkez - R @ merkez + t      # parcanin kendi merkezi etrafinda don + otele
    return T


def uygula(T, P):
    return P @ T[:3, :3].T + T[:3, 3]


def pcd(P):
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(np.asarray(P, dtype=np.float64))
    return p


def hata_olc(E, temiz):
    """E = T_est * T_pert (birim olmali). Noktalarin gercek konumdan sapmasi ve donme hatasi."""
    sapma = np.linalg.norm(uygula(E, temiz) - temiz, axis=1)
    aci = np.degrees(np.arccos(np.clip((np.trace(E[:3, :3]) - 1) / 2, -1, 1)))
    return float(np.sqrt(np.mean(sapma ** 2))), float(aci), sapma


def ortusme(kaynak, hedef, esik=0.3):
    d, _ = cKDTree(hedef).query(kaynak, distance_upper_bound=esik)
    return float(np.mean(np.isfinite(d)))


def stitcher(**kw):
    return PointCloudStitcher(voxel_size=0.4, seed=0, **kw)


# ───────────────────────── deneyler ─────────────────────────

def deney_cift(hastalar, sigma, seviyeler, yontemler, tohum=0):
    satirlar = []
    for h, parcalar in hastalar.items():
        for i in range(len(parcalar) - 1):
            (n_t, f_t), (n_s, f_s) = parcalar[i], parcalar[i + 1]
            ciftno = sabit_hash(h, n_t, n_s) % 100000
            hedef_temiz, hedef = tarama_simule_et(f_t, tohum * 1000 + ciftno * 2, sigma)
            kaynak_temiz, kaynak = tarama_simule_et(f_s, tohum * 1000 + ciftno * 2 + 1, sigma)
            ort = ortusme(kaynak_temiz, hedef_temiz)
            for sev in seviyeler:
                rng = np.random.RandomState((ciftno * 7 + sabit_hash(sev)) % 2 ** 31)
                T_pert = rastgele_hareket(rng, sev, kaynak_temiz.mean(0))
                kaynak_hareketli = uygula(T_pert, kaynak)
                for ad in yontemler:
                    st = stitcher(**YONTEMLER[ad])
                    t0 = time.time()
                    try:
                        T_est = st.register(pcd(kaynak_hareketli), pcd(hedef))
                        rmse, aci, _ = hata_olc(T_est @ T_pert, kaynak_temiz)
                    except Exception as e:
                        kayitci.warning(f"{h} {n_t}-{n_s} {ad}: {e}")
                        rmse, aci = float("inf"), float("inf")
                    satirlar.append(dict(hasta=h, hedef=n_t, kaynak=n_s, ortusme=ort, sigma=sigma, hareket=sev,
                                         yontem=ad, rmse_mm=rmse, donme_hata_derece=aci,
                                         basarili=bool(rmse < BASARI_ESIGI_MM), sure_sn=time.time() - t0,
                                         fitness=st.last_result.get("fitness"), inlier_rmse=st.last_result.get("inlier_rmse"),
                                         ransac=st.last_result.get("ransac_used"), gecerli=st.last_result.get("valid")))
        kayitci.info(f"[cift] {h} bitti (sigma={sigma}, {len(satirlar)} kayit)")
    return satirlar


def deney_zincir(hastalar, sigma=0.02, seviye="kucuk", tohum=0):
    """Tam ark birlestirme. Ilk parca referans (dogru konumda); digerleri sirayla eklenir."""
    satirlar, ornek = [], {}
    for h, parcalar in hastalar.items():
        rng = np.random.RandomState(sabit_hash(h) % 2 ** 31)
        taramalar = []
        for k, (n, f) in enumerate(parcalar):
            temiz, gurultulu = tarama_simule_et(f, tohum * 7919 + k + sabit_hash(h) % 10000, sigma)
            T_pert = np.eye(4) if k == 0 else rastgele_hareket(rng, seviye, temiz.mean(0))
            taramalar.append((n, temiz, uygula(T_pert, gurultulu), T_pert))

        for strateji, mod in (("kare→model", "global"), ("kare→kare", "global"), ("kare→model (hibrit)", "hybrid")):
            st = stitcher(icp_method="point_to_plane", registration_mode=mod)
            T_glob_onceki = np.eye(4)
            renkli = []
            for k, (n, temiz, hareketli, T_pert) in enumerate(taramalar):
                if k == 0:
                    st.align_and_merge(pcd(hareketli))
                    T_est = np.eye(4)
                elif strateji.startswith("kare→model"):
                    T_est = st.align_and_merge(pcd(hareketli))
                else:
                    T_rel = st.register(pcd(hareketli), pcd(taramalar[k - 1][2]))
                    T_est = T_glob_onceki @ T_rel
                T_glob_onceki = T_est
                rmse, aci, sapma = hata_olc(T_est @ T_pert, temiz)
                satirlar.append(dict(hasta=h, strateji=strateji, sira=k, dis=n, rmse_mm=rmse,
                                     ransac=st.last_result.get("ransac_used") if k else False,
                                     donme_hata_derece=aci, sigma=sigma, hareket=seviye))
                renkli.append((uygula(T_est @ T_pert, temiz), sapma))
            ornek[(h, strateji)] = renkli
        kayitci.info(f"[zincir] {h} bitti")
    return satirlar, ornek


# ───────────────────────── raporlama ─────────────────────────

def ozet_ve_sekiller(A, B, C, ornek, cikti, sekil_dizin):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    os.makedirs(sekil_dizin, exist_ok=True)
    A, B, C = pd.DataFrame(A), pd.DataFrame(B), pd.DataFrame(C)
    renk = {"ICP (nokta-nokta)": "#b0b0b0", "ICP (nokta-düzlem)": "#e39b3b", "RANSAC + ICP (nokta-düzlem)": "#7fa7cf",
            "Hibrit (ICP → gerekirse RANSAC)": "#1f4e79"}
    sev_ad = {"kucuk": "Küçük\n(2–5°, 1–3 mm)", "orta": "Orta\n(10–15°, 4–6 mm)", "buyuk": "Büyük\n(25–35°, 8–12 mm)"}

    # Tablo A
    tA = (A.groupby(["hareket", "yontem"])
            .agg(cift=("rmse_mm", "size"), basari_orani=("basarili", "mean"),
                 medyan_rmse_mm=("rmse_mm", "median"),
                 basarili_ort_rmse_mm=("rmse_mm", lambda s: s[s < BASARI_ESIGI_MM].mean()),
                 medyan_donme_hata=("donme_hata_derece", "median"), ort_sure_sn=("sure_sn", "mean"))
            .reset_index())
    tA["hareket"] = pd.Categorical(tA["hareket"], ["kucuk", "orta", "buyuk"])
    tA = tA.sort_values(["hareket", "yontem"])
    tA.to_csv(os.path.join(cikti, "tablo_A_cift.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 4.4))
    x = np.arange(3); w = 0.2
    for i, y in enumerate(YONTEMLER):
        v = [tA[(tA.hareket == s) & (tA.yontem == y)].basari_orani.values[0] for s in ["kucuk", "orta", "buyuk"]]
        ax.bar(x + (i - 1.5) * w, v, w, color=renk[y], label=y)
        for xi, vi in zip(x + (i - 1.5) * w, v):
            ax.text(xi, vi + 0.015, f"%{vi * 100:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([sev_ad[s] for s in ["kucuk", "orta", "buyuk"]])
    ax.set_ylim(0, 1.12); ax.set_ylabel(f"Başarı oranı (hata < {BASARI_ESIGI_MM} mm)")
    ax.set_title(f"Parça çifti hizalama başarısı ({A[A.yontem == list(YONTEMLER)[0]].groupby('hareket').size().iloc[0]} komşu parça çifti, {A.hasta.nunique()} hasta)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(sekil_dizin, "sekil7_birlestirme_basari.png"), dpi=200); plt.close(fig)

    # Ortusme - basari (RANSAC+ICP, buyuk hareket)
    alt = A[(A.yontem == "Hibrit (ICP → gerekirse RANSAC)")]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for s, mk in [("kucuk", "o"), ("orta", "s"), ("buyuk", "^")]:
        d = alt[alt.hareket == s]
        ax.scatter(d.ortusme * 100, np.clip(d.rmse_mm, 1e-3, 50), s=14, marker=mk, alpha=.7, label=sev_ad[s].replace("\n", " "))
    ax.axhline(BASARI_ESIGI_MM, color="#c0392b", ls="--", lw=1, label=f"Başarı eşiği {BASARI_ESIGI_MM} mm")
    ax.set_yscale("log"); ax.set_xlabel("Örtüşme oranı (%)"); ax.set_ylabel("Konum hatası RMSE (mm, log)")
    ax.set_title("Hibrit yöntem: örtüşme ve hata ilişkisi"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(sekil_dizin, "sekil8_ortusme_hata.png"), dpi=200); plt.close(fig)

    # Tablo B + drift
    tB = (B[B.sira > 0].groupby("strateji")
            .agg(parca=("rmse_mm", "size"), ort_rmse_mm=("rmse_mm", "mean"), medyan_rmse_mm=("rmse_mm", "median"),
                 maks_rmse_mm=("rmse_mm", "max"), basari_orani=("rmse_mm", lambda s: (s < BASARI_ESIGI_MM).mean()))
            .reset_index())
    son = B.loc[B.groupby(["hasta", "strateji"]).sira.idxmax()].groupby("strateji").rmse_mm.agg(["mean", "median", "max"])
    tB = tB.merge(son.rename(columns={"mean": "son_parca_ort_mm", "median": "son_parca_medyan_mm", "max": "son_parca_maks_mm"}),
                  left_on="strateji", right_index=True)
    tB.to_csv(os.path.join(cikti, "tablo_B_zincir.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    for s, c in [("kare→kare", "#e39b3b"), ("kare→model", "#7fa7cf"), ("kare→model (hibrit)", "#1f4e79")]:
        d = B[(B.strateji == s) & (B.sira > 0)].groupby("sira").rmse_mm
        med, q1, q3 = d.median(), d.quantile(.25), d.quantile(.75)
        ax.plot(med.index, med.values, color=c, marker="o", ms=3, label=f"{s} (medyan)")
        ax.fill_between(med.index, q1.values, q3.values, color=c, alpha=.15)
    ax.set_xlabel("Eklenen parça sırası (ark boyunca)"); ax.set_ylabel("Konum hatası RMSE (mm)")
    ax.set_yscale("log"); ax.set_ylabel("Konum hatası RMSE (mm, log)")
    ax.set_title("Tam ark birleştirmede biriken hata (gölge: çeyrekler arası)"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(sekil_dizin, "sekil9_drift.png"), dpi=200); plt.close(fig)

    # Isi haritasi: bir hasta, iki strateji
    if ornek:
        def ort(h, s):
            return np.concatenate([e for _, e in ornek[(h, s)]]).mean()
        h = max({k[0] for k in ornek}, key=lambda h: ort(h, "kare→model") - ort(h, "kare→model (hibrit)"))
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        vmax = 0.2
        for ax, s in zip(axs, ("kare→model", "kare→model (hibrit)")):
            P = np.concatenate([p for p, _ in ornek[(h, s)]]); e = np.concatenate([e for _, e in ornek[(h, s)]])
            o = np.argsort(P[:, 2]); sc = ax.scatter(P[o, 0], P[o, 1], c=e[o], s=0.3, cmap="viridis", vmin=0, vmax=vmax, rasterized=True)
            ax.set_aspect("equal"); ax.axis("off")
            ax.set_title(f"{'Her karede RANSAC' if s == 'kare→model' else 'Hibrit (ICP takip + gerekirse RANSAC)'}\n"
                         f"ortalama sapma {e.mean():.3f} mm, %95: {np.percentile(e, 95):.3f} mm")
        cb = fig.colorbar(sc, ax=axs, fraction=.025, pad=.02, extend="max"); cb.set_label("Gerçek konumdan sapma (mm)")
        fig.suptitle(f"Birleştirilmiş tam ark — sapma ısı haritası (hasta {h}, {len(ornek[(h, 'kare→model')])} parça)")
        fig.savefig(os.path.join(sekil_dizin, "sekil10_sapma_haritasi.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

    # Tablo C + grafik
    tC = (C.groupby(["sigma", "hareket"]).agg(basari_orani=("basarili", "mean"), medyan_rmse_mm=("rmse_mm", "median"),
                                             p90_rmse_mm=("rmse_mm", lambda s: s.quantile(.9))).reset_index())
    tC.to_csv(os.path.join(cikti, "tablo_C_gurultu.csv"), index=False)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for s, c in [("kucuk", "#2f6ea5"), ("buyuk", "#c0392b")]:
        d = tC[tC.hareket == s]
        ax.plot(d.sigma, d.medyan_rmse_mm, marker="o", color=c, label=f"{sev_ad[s].replace(chr(10), ' ')} — medyan")
        ax.plot(d.sigma, d.p90_rmse_mm, ls=":", color=c, lw=1, label=f"{sev_ad[s].replace(chr(10), ' ')} — %90'lık dilim")
    ax.axhline(BASARI_ESIGI_MM, color="#888", ls="--", lw=1)
    ax.set_yscale("log"); ax.set_xlabel("Sensör gürültüsü σ (mm)"); ax.set_ylabel("Konum hatası RMSE (mm, log)")
    ax.set_title("Gürültü duyarlılığı (hibrit yöntem)"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(sekil_dizin, "sekil11_gurultu.png"), dpi=200); plt.close(fig)
    return tA, tB, tC


def main():
    p = argparse.ArgumentParser(description="Parca tarama birlestirme deneyi")
    p.add_argument("--data", default="data/four/partial_scans")
    p.add_argument("--out", default="runs/birlestirme")
    p.add_argument("--figs", default="rapor_gorselleri")
    p.add_argument("--max-patients", type=int, default=12)
    p.add_argument("--sigma", type=float, default=0.02)
    p.add_argument("--skip", nargs="*", default=[], choices=["A", "B", "C"])
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    hastalar = dict(list(hasta_parcalari(a.data).items())[:a.max_patients])
    kayitci.info(f"{len(hastalar)} hasta, {sum(len(v) for v in hastalar.values())} parca")

    def kaydet(ad, veri):
        with open(os.path.join(a.out, ad), "w", encoding="utf-8") as f:
            json.dump(veri, f, ensure_ascii=False)

    def yukle(ad):
        yol = os.path.join(a.out, ad)
        return json.load(open(yol, encoding="utf-8")) if os.path.exists(yol) else None

    A = yukle("A.json") if "A" in a.skip else None
    if A is None:
        A = deney_cift(hastalar, a.sigma, ["kucuk", "orta", "buyuk"], list(YONTEMLER)); kaydet("A.json", A)
    B, ornek = (yukle("B.json"), {}) if "B" in a.skip else (None, None)
    if B is None:
        B, ornek = deney_zincir(hastalar, a.sigma); kaydet("B.json", B)
    C = yukle("C.json") if "C" in a.skip else None
    if C is None:
        C = []
        for s in [0.0, 0.02, 0.05, 0.1, 0.2]:
            C += deney_cift(hastalar, s, ["kucuk", "buyuk"], ["Hibrit (ICP → gerekirse RANSAC)"])
        kaydet("C.json", C)

    tA, tB, tC = ozet_ve_sekiller(A, B, C, ornek, a.out, a.figs)
    print(tA.to_string(index=False)); print(tB.to_string(index=False)); print(tC.to_string(index=False))


if __name__ == "__main__":
    main()
