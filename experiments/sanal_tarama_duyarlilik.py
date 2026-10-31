"""
Sanal tarama duyarlilik analizi:
  1) Kalibrasyon hatasi: odak uzakligi, lazer duzlemi acisi, baz uzunlugu
     (goruntuler dogru parametrelerle uretilir; hat hatali parametrelerle calisir -> yeniden render gerekmez)
  2) Goruntu gurultusu: kamera gurultusu sigma (yeniden render)
  3) IMU kaymasi: kare basina egilim (bias) -- sadece IMU ve IMU + hibrit ICP
Olcut: 3D noktalarin gercek yuzeye uzakligi (mm). Sekil: sekil16_duyarlilik.png
"""
import os
import sys
import json
import argparse
import numpy as np

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)
sys.path.insert(0, os.path.dirname(__file__))
import sanal_tarama_deneyi as sd
from ai_model.dataset import obj_oku, mesh_oku


def nokta_hatasi(kayitlar, K, lazer, sahne, mod="ideal", imu=None, alt_ornek=4):
    k = [x for i, x in enumerate(kayitlar) if i % alt_ornek == 0] if mod == "ideal" else kayitlar
    k = sd.uc_boyuta([dict(x) for x in k], K, lazer)
    if mod != "ideal":
        k = sd.imu_pozlari(k, **(imu or {}))
    P, _ = sd.birlestir(k, mod)
    d = sd.uzaklik(sahne, P)
    ph = sd.poz_hatasi(k)
    return dict(medyan=float(np.median(d)), rms=float(np.sqrt((d ** 2).mean())), p95=float(np.percentile(d, 95)),
                poz_ort=float(ph.mean()))


def hatali_parametre(K, lazer, odak=0.0, aci=0.0, baz=0.0):
    K = K.copy(); K[0, 0] *= 1 + odak; K[1, 1] *= 1 + odak
    a = np.arctan2(lazer[2], lazer[1]) + np.radians(aci)
    b = -lazer[3] / lazer[1] + baz
    return K, np.array([0.0, np.cos(a), np.sin(a), -b * np.cos(a)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", required=True)
    p.add_argument("--out", default="runs/sanal_tarama")
    p.add_argument("--figs", default="rapor_gorselleri")
    p.add_argument("--skip-noise", action="store_true")
    a = p.parse_args()
    v, f = obj_oku(a.mesh)
    _, _, etiket, _ = mesh_oku(a.mesh, os.path.splitext(a.mesh)[0] + ".json")
    ad = os.path.splitext(os.path.basename(a.mesh))[0]
    cikti = os.path.join(a.out, ad)
    sahne = sd.gt_sahnesi(v, f)
    kayit, K, L = sd.kareleri_yukle(cikti)
    sonuc = {"kalibrasyon": [], "gurultu": [], "imu": []}

    for tur, degerler in [("odak", [-0.02, -0.01, -0.005, 0, 0.005, 0.01, 0.02]),
                          ("aci", [-1.0, -0.5, -0.25, 0, 0.25, 0.5, 1.0]),
                          ("baz", [-0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2])]:
        for d in degerler:
            Kh, Lh = hatali_parametre(K, L, **{tur: d})
            r = nokta_hatasi(kayit, Kh, Lh, sahne)
            sonuc["kalibrasyon"].append(dict(parametre=tur, hata=d, **r))
            print("kalibrasyon", tur, d, r, flush=True)

    for bias in [0.0, 0.001, 0.002, 0.005, 0.01]:
        imu = dict(sapma_mm_kare=bias)
        for mod in ("imu_ham", "imu"):
            r = nokta_hatasi(kayit, K, L, sahne, mod, imu)
            sonuc["imu"].append(dict(bias_mm_kare=bias, mod=mod, **r))
            print("imu", bias, mod, r, flush=True)

    if not a.skip_noise:
        for sigma in [0.0, 3.0, 10.0, 25.0, 50.0]:
            dosya = f"kareler_gurultu_{int(sigma)}.npz"
            if not os.path.exists(os.path.join(cikti, dosya)):
                sd.tara(v, f, etiket, cikti, goruntu_gurultusu=sigma, dosya=dosya, tohum=1)
            kg, Kg, Lg = sd.kareleri_yukle(cikti, dosya)
            r = nokta_hatasi(kg, Kg, Lg, sahne)
            r["nokta_sayisi_oran"] = float(sum(len(x["tepeler"]) for x in kg) / max(1, sum(len(x["tepeler"]) for x in kayit)))
            sonuc["gurultu"].append(dict(sigma=sigma, **r))
            print("gurultu", sigma, r, flush=True)

    with open(os.path.join(cikti, "duyarlilik.json"), "w", encoding="utf-8") as fh:
        json.dump(sonuc, fh, indent=1)
    ciz(sonuc, a.figs)


def ciz(sonuc, figs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    n = 3 if sonuc["gurultu"] else 2
    fig, axs = plt.subplots(1, n, figsize=(5.2 * n, 4))
    ax = axs[0]
    ad = {"odak": ("Odak uzaklığı hatası", 100, "%"), "aci": ("Lazer düzlemi açı hatası", 1, "°"),
          "baz": ("Kamera–lazer baz hatası", 1, " mm")}
    for tur, c in [("odak", "#2f6ea5"), ("aci", "#c0392b"), ("baz", "#e39b3b")]:
        d = [r for r in sonuc["kalibrasyon"] if r["parametre"] == tur]
        x = np.arange(len(d)) - len(d) // 2
        ax.plot(x, [r["rms"] for r in d], marker="o", color=c,
                label=f"{ad[tur][0]} (adım: {', '.join(f'{r['hata'] * ad[tur][1]:g}' for r in d if r['hata'] > 0)}{ad[tur][2]})")
    ax.set_xticks(range(-3, 4)); ax.set_xticklabels(["−3", "−2", "−1", "0", "+1", "+2", "+3"])
    ax.set_xlabel("Hata seviyesi (adım)"); ax.set_ylabel("Nokta hatası RMS (mm)")
    ax.set_title("Kalibrasyon hatasına duyarlılık"); ax.legend(frameon=False, fontsize=7.5, loc="upper center")
    ax.set_ylim(0, None)

    ax = axs[1]
    for mod, c, lab in [("imu_ham", "#c0392b", "Sadece IMU"), ("imu", "#1f4e79", "IMU + hibrit ICP (kapılı)")]:
        d = [r for r in sonuc["imu"] if r["mod"] == mod]
        ax.plot([r["bias_mm_kare"] * 1000 for r in d], [r["poz_ort"] for r in d], marker="o", color=c, label=lab)
    ax.set_xlabel("IMU kayma eğilimi (µm / kare)"); ax.set_ylabel("Ortalama poz hatası (mm)")
    ax.set_yscale("log"); ax.set_ylabel("Ortalama poz hatası (mm, log)")
    ax.set_title("IMU kaymasına duyarlılık\n(kapı: 3 mm — daha büyük kaymada ICP düzeltmesi reddedilir)", fontsize=10); ax.legend(frameon=False)
    if n == 3:
        ax = axs[2]
        d = sonuc["gurultu"]
        ax.plot([r["sigma"] for r in d], [r["rms"] for r in d], marker="o", color="#2f6ea5", label="Nokta hatası RMS")
        ax.plot([r["sigma"] for r in d], [r["medyan"] for r in d], marker="s", color="#7fa7cf", label="Medyan")
        ax2 = ax.twinx()
        ax2.plot([r["sigma"] for r in d], [r["nokta_sayisi_oran"] * 100 for r in d], ls=":", color="#888", label="Tespit edilen nokta (%)")
        ax2.set_ylabel("Tespit edilen nokta (%, σ=3'e göre)", color="#666"); ax2.spines["top"].set_visible(False)
        ax.set_yscale("log"); ax.set_xlabel("Kamera gürültüsü σ (0–255 ölçeği)"); ax.set_ylabel("Nokta hatası (mm, log)")
        ax.set_title("Görüntü gürültüsüne duyarlılık"); ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(figs, "sekil16_duyarlilik.png"), dpi=200); plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--sadece-ciz":
        ciz(json.load(open(sys.argv[2])), sys.argv[3])
    else:
        main()
