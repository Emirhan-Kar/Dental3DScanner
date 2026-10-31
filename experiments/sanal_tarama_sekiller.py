"""Sanal tarama deneyi icin rapor sekilleri (sekil12-15)."""
import os
import sys
import json
import argparse
import numpy as np
import open3d as o3d
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)
from ai_model.dataset import obj_oku, mesh_oku
from ui.viewer import get_fdi_color_palette

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def ust_gorunum(ax, V, renk, yukari_z=True, s=0.15, **kw):
    o = np.argsort(V[:, 2] if yukari_z else -V[:, 2])
    sc = ax.scatter(V[o, 0], V[o, 1], c=renk[o] if renk.ndim > 1 else renk[o], s=s, linewidths=0, rasterized=True, **kw)
    ax.set_aspect("equal"); ax.axis("off")
    return sc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", required=True)
    p.add_argument("--run", required=True, help="runs/sanal_tarama/<id>")
    p.add_argument("--figs", default="rapor_gorselleri")
    a = p.parse_args()
    os.makedirs(a.figs, exist_ok=True)
    v, f = obj_oku(a.mesh)
    _, _, etiket, _ = mesh_oku(a.mesh, os.path.splitext(a.mesh)[0] + ".json")
    alt = v[etiket > 0, 2].mean() > v[etiket == 0, 2].mean()
    R = json.load(open(os.path.join(a.run, "sonuc_ideal.json")))
    palet = get_fdi_color_palette()

    # 12: ornek kare + tespit, eski vs yeni dedektor (ayni kare, iki cizginin karsilastigi bolge buyutulmus)
    yeni = cv2.imread(os.path.join(a.run, "ornek_kare_kareler_1.png"))
    eski = cv2.imread(os.path.join(a.run, "ornek_kare_kareler_eski_dedektor_1.png"))
    if yeni is not None and eski is not None:
        w = yeni.shape[1] // 2
        ham = yeni[:, :w]
        # en cok fark olan bolgeyi bul: iki gorselin isaretli (yesil) piksel farki
        fark = (np.abs(eski[:, w:].astype(int) - yeni[:, w:].astype(int)).sum(2) > 60).astype(np.float32)
        fark = cv2.GaussianBlur(fark, (0, 0), 15)
        cy, cx = np.unravel_index(np.argmax(fark), fark.shape)
        y0, x0 = int(np.clip(cy - 55, 0, ham.shape[0] - 110)), int(np.clip(cx - 80, 0, w - 160))
        kes = lambda im: cv2.resize(im[y0:y0 + 110, x0:x0 + 160], (480, 330), interpolation=cv2.INTER_NEAREST)
        fig, axs = plt.subplots(1, 3, figsize=(16, 5.6), gridspec_kw={"width_ratios": [1.3, 1, 1]})
        g = ham.copy(); cv2.rectangle(g, (x0, y0), (x0 + 160, y0 + 110), (0, 255, 255), 2)
        for ax, im, t in zip(axs, [g, kes(eski[:, w:]), kes(yeni[:, w:])],
                             ["Sanal kamera görüntüsü (640×480)\n(sarı kutu: büyütülen bölge)",
                              "Eski dedektör (3× büyütme)\nsütundaki tüm lazer piksellerinin ağırlık merkezi\n→ iki çizgi arasına sahte noktalar",
                              "Yeni dedektör (3× büyütme)\nsütun başına en güçlü bağlantılı parça"]):
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); ax.set_title(t, fontsize=10); ax.axis("off")
        fig.tight_layout(); fig.savefig(os.path.join(a.figs, "sekil12_sanal_kamera_tespit.png"), dpi=180); plt.close(fig)

    # 13: sapma isi haritasi + kapsama
    rec = o3d.io.read_triangle_mesh(os.path.join(a.run, "rekonstruksiyon_ideal.ply"))
    Vr = np.asarray(rec.vertices)
    d_mesh = np.load(os.path.join(a.run, "sapma_mesh_ideal.npy"))
    d_kap = np.load(os.path.join(a.run, "kapsama_ideal.npy"))
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.6))
    sc = ust_gorunum(axs[0], Vr, d_mesh, alt, cmap="viridis", vmin=0, vmax=0.1)
    cb = fig.colorbar(sc, ax=axs[0], fraction=.035, pad=.01, extend="max"); cb.set_label("Gerçek yüzeyden sapma (mm)")
    md = R["mesh_dogrulugu_mm"]
    axs[0].set_title(f"Yeniden oluşturulan yüzey — sapma\nmedyan {md['medyan']:.3f} mm, RMS {md['rms']:.3f} mm, %95: {md['p95']:.3f} mm")
    renk = np.where((d_kap < 0.2)[:, None], [[0.75, 0.75, 0.78]], [[0.85, 0.15, 0.15]])
    renk[(etiket > 0) & (d_kap < 0.2)] = [0.2, 0.55, 0.3]
    ust_gorunum(axs[1], v, renk, alt)
    axs[1].set_title(f"Kapsama (gerçek model üzerinde)\nyeşil: taranan diş yüzeyi (%{R['kapsama_dis_0.2mm'] * 100:.1f}), "
                     f"kırmızı: taranamayan, gri: taranan diş eti")
    fig.tight_layout(); fig.savefig(os.path.join(a.figs, "sekil13_sanal_tarama_sapma_kapsama.png"), dpi=200); plt.close(fig)

    # 14: segmentasyon (yeniden olusturulan mesh)
    tp = os.path.join(a.run, "seg_tahmin_ideal.npy")
    if os.path.exists(tp):
        tahmin, hedef = np.load(tp), np.load(os.path.join(a.run, "seg_hedef_ideal.npy"))
        fig, axs = plt.subplots(1, 2, figsize=(13, 5.6))
        ust_gorunum(axs[0], Vr, palet[hedef], alt); axs[0].set_title("Gerçek etiketler (orijinal modelden aktarıldı)")
        ust_gorunum(axs[1], Vr, palet[tahmin], alt)
        axs[1].set_title(f"DGCNN tahmini — sanal taramadan elde edilen model\nmIoU (dişler) {R['segmentasyon']['mIoU_dis']:.3f}")
        fig.tight_layout(); fig.savefig(os.path.join(a.figs, "sekil14_sanal_tarama_segmentasyon.png"), dpi=200); plt.close(fig)

    # 15: uctan uca akis (4 panel)
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.6))
    if yeni is not None:
        axs[0].imshow(cv2.cvtColor(yeni[:, :yeni.shape[1] // 2], cv2.COLOR_BGR2RGB)); axs[0].axis("off")
    axs[0].set_title("1. Kamera görüntüsü + lazer\n(1.461 kare, 3 geçiş)")
    kay = np.load(os.path.join(a.run, "kareler.npz"))
    from processing.triangulation import Triangulator
    tri = Triangulator(camera_matrix=kay["K"], laser_plane=kay["lazer"])
    sin = np.concatenate([[0], np.cumsum(kay["tepe_sayisi"])])
    P = []
    for i in range(0, len(kay["gecis"]), 3):
        pk = kay["tepeler"][sin[i]:sin[i + 1]]
        if len(pk):
            Pc = tri.pixels_to_3d(pk); T = kay["T"][i]; P.append(Pc @ T[:3, :3].T + T[:3, 3])
    P = np.concatenate(P)
    ust_gorunum(axs[1], P, np.full(len(P), 0.0), alt, s=0.05, cmap="Greys", vmin=-1, vmax=1)
    axs[1].set_title(f"2. Lazer tespiti + üçgenleme\n({R['nokta_sayisi'] / 1e3:.0f} bin 3D nokta)")
    ust_gorunum(axs[2], Vr, d_mesh, alt, cmap="viridis", vmin=0, vmax=0.1)
    axs[2].set_title(f"3. Poisson yüzey\nRMS sapma {md['rms']:.3f} mm")
    if os.path.exists(tp):
        ust_gorunum(axs[3], Vr, palet[tahmin], alt)
        axs[3].set_title(f"4. DGCNN diş segmentasyonu\nmIoU {R['segmentasyon']['mIoU_dis']:.3f}")
    for ax in axs[1:]:
        ax.axis("off")
    fig.suptitle("Uçtan uca sanal tarama — test setinden bir hasta (modelin eğitimde görmediği)", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(a.figs, "sekil15_uctan_uca.png"), dpi=200); plt.close(fig)
    # 17: poz senaryolari karsilastirmasi
    senaryolar = [("ideal", "İdeal poz (bilinen)"), ("imu_ham", "Sadece IMU (kaymalı)"), ("imu", "IMU + hibrit ICP + makullük kapısı")]
    if all(os.path.exists(os.path.join(a.run, f"sonuc_{m}.json")) for m, _ in senaryolar):
        fig, axs = plt.subplots(2, 3, figsize=(17, 10.5))
        for j, (m, t) in enumerate(senaryolar):
            Rm = json.load(open(os.path.join(a.run, f"sonuc_{m}.json")))
            Vm = np.asarray(o3d.io.read_triangle_mesh(os.path.join(a.run, f"rekonstruksiyon_{m}.ply")).vertices)
            dp = os.path.join(a.run, f"sapma_mesh_{m}_hizali.npy")
            dm = np.load(dp if os.path.exists(dp) else os.path.join(a.run, f"sapma_mesh_{m}.npy"))
            acc = Rm.get("en_iyi_hizalama_sonrasi", Rm)["mesh_dogrulugu_mm"]
            sc = ust_gorunum(axs[0, j], Vm, dm, alt, cmap="viridis", vmin=0, vmax=0.5)
            axs[0, j].set_title(f"{t}\npoz hatası ort. {Rm['poz_hatasi_ort_mm']:.2f} mm | yüzey RMS {acc['rms']:.3f} mm")
            tp_m = os.path.join(a.run, f"seg_tahmin_{m}.npy")
            if os.path.exists(tp_m):
                ust_gorunum(axs[1, j], Vm, palet[np.load(tp_m)], alt)
                axs[1, j].set_title(f"DGCNN segmentasyonu — mIoU {Rm['segmentasyon']['mIoU_dis']:.3f}")
        fig.subplots_adjust(left=0.01, right=0.9, top=0.9, bottom=0.02, wspace=0.05, hspace=0.12)
        cax = fig.add_axes([0.92, 0.55, 0.012, 0.3])
        cb = fig.colorbar(sc, cax=cax, extend="max"); cb.set_label("Sapma (mm, en iyi hizalama sonrası)")
        fig.suptitle("Probun konumu bilinmediğinde: IMU kayması ve ICP düzeltmesinin etkisi", fontsize=13)
        fig.savefig(os.path.join(a.figs, "sekil17_poz_senaryolari.png"), dpi=170, bbox_inches="tight"); plt.close(fig)
    print("ok")


if __name__ == "__main__":
    main()
