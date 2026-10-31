"""
Uctan Uca Sanal Tarama Deneyi

  Teeth3DS cenesi --(sanal kamera + lazer)--> goruntuler
      --> LaserDetector (alt-piksel cizgi tespiti)
      --> Triangulator (piksel -> 3D, kamera koordinati)
      --> poz: 'ideal' (bilinen) veya 'imu' (kaymali IMU + yama bazli hibrit ICP duzeltmesi)
      --> filtreleme + Poisson mesh (MeshGenerator)
      --> DGCNN dis segmentasyonu
  ve her adim gercek cene modeliyle (ground truth) karsilastirilir.

Kullanim:
  python experiments/sanal_tarama_deneyi.py --mesh data/3DTeethData/.../VBUD4E5R_lower.obj --out runs/sanal_tarama
  (goruntuler bir kez uretilir ve onbellege alinir: <out>/<id>/kareler.npz)
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import open3d as o3d
import cv2

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from ai_model.dataset import mesh_oku, obj_oku, FDI_SINIFLARI, kose_normalleri
from simulation.sanal_tarayici import SanalTarayici, SanalKamera, ark_egrisi, tarama_pozlari
from processing.laser_detector import LaserDetector
from processing.triangulation import Triangulator
from processing.icp_stitcher import PointCloudStitcher
from processing.mesh_generator import MeshGenerator
from utils.logger import get_logger

kayitci = get_logger("sanal_tarama")

GECISLER = [("tepe", 0.0), ("yanak", 35.0), ("dil", -35.0)]


# ───────────────────────── 1. goruntu uretimi + tespit ─────────────────────────

def tara(v, f, etiket, cikti, adim_mm=0.25, goruntu_gurultusu=3.0, ornek_kare_sayisi=4, tohum=0,
         dosya="kareler.npz", dedektor=None):
    """Tum gecisleri render eder, her karede lazer tespiti yapar; tepe noktalarini kaydeder."""
    st = SanalTarayici(v, f, etiket, goruntu_gurultusu=goruntu_gurultusu, tohum=tohum)
    kam = st.kamera
    det = dedektor or LaserDetector()
    alt_cene = v[etiket > 0, 2].mean() > v[etiket == 0, 2].mean()
    yukari = np.array([0, 0, 1.0]) if alt_cene else np.array([0, 0, -1.0])
    ark, teget = ark_egrisi(v, etiket, FDI_SINIFLARI, adim_mm, yukari=yukari)

    kayitlar, ornekler = [], []
    t0 = time.time()
    for g, (ad, egim) in enumerate(GECISLER):
        pozlar = tarama_pozlari(ark, teget, yukari, egim_derece=egim)
        for k, T in enumerate(pozlar):
            img, bilgi = st.kare_uret(T)
            tepeler, _ = det.detect_subpixel_peaks(img)
            kayitlar.append(dict(gecis=g, sira=k, T=T, tepeler=np.array(tepeler, dtype=np.float32).reshape(-1, 2)))
            if k in np.linspace(len(pozlar) * 0.2, len(pozlar) * 0.8, ornek_kare_sayisi).astype(int) and g == 0:
                ornekler.append((img, np.array(tepeler)))
        kayitci.info(f"Gecis '{ad}' bitti: {len(pozlar)} kare ({time.time() - t0:.0f} sn)")

    np.savez_compressed(os.path.join(cikti, dosya),
                        gecis=np.array([k["gecis"] for k in kayitlar]),
                        sira=np.array([k["sira"] for k in kayitlar]),
                        T=np.stack([k["T"] for k in kayitlar]),
                        tepe_sayisi=np.array([len(k["tepeler"]) for k in kayitlar]),
                        tepeler=np.concatenate([k["tepeler"] for k in kayitlar]),
                        K=kam.K, lazer=kam.lazer_duzlemi)
    for i, (img, tp) in enumerate(ornekler):
        vis = img.copy()
        for u, vv in tp[::2]:
            cv2.circle(vis, (int(round(u)), int(round(vv))), 1, (0, 255, 0), -1)
        cv2.imwrite(os.path.join(cikti, f"ornek_kare_{os.path.splitext(dosya)[0]}_{i}.png"), np.hstack([img, vis]))
    return kayitlar


def kareleri_yukle(cikti, dosya="kareler.npz"):
    with np.load(os.path.join(cikti, dosya)) as z:
        d = {k: z[k] for k in z.files}          # her diziyi bir kez ac (npz her erisimde yeniden acar)
    sinirlar = np.concatenate([[0], np.cumsum(d["tepe_sayisi"])])
    kayitlar = [dict(gecis=int(d["gecis"][i]), sira=int(d["sira"][i]), T=d["T"][i],
                     tepeler=d["tepeler"][sinirlar[i]:sinirlar[i + 1]]) for i in range(len(d["gecis"]))]
    return kayitlar, d["K"], d["lazer"]


# ───────────────────────── 2. 3D noktalar + poz ─────────────────────────

def uc_boyuta(kayitlar, K, lazer):
    tri = Triangulator(camera_matrix=K, laser_plane=lazer)
    for k in kayitlar:
        k["P_kam"] = tri.pixels_to_3d(k["tepeler"]).astype(np.float64) if len(k["tepeler"]) else np.empty((0, 3))
    return kayitlar


def _rot(rng, sigma_derece):
    """Kucuk rastgele donme (Rodrigues; scipy gerektirmez)."""
    return o3d.geometry.get_rotation_matrix_from_axis_angle(np.radians(rng.normal(0, sigma_derece, 3)))


def en_yakin(referans, sorgu):
    """Her sorgu noktasi icin referanstaki en yakin noktanin (uzaklik, indeks) degeri.
    Open3D tensor NNS kullanir (scipy gerektirmez)."""
    nns = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(np.ascontiguousarray(referans, dtype=np.float32)))
    nns.knn_index()
    idx, d2 = nns.knn_search(o3d.core.Tensor(np.ascontiguousarray(sorgu, dtype=np.float32)), 1)
    return np.sqrt(d2.numpy()[:, 0].astype(np.float64)), idx.numpy()[:, 0].astype(np.int64)


def imu_pozlari(kayitlar, sigma_rot_derece=0.02, sigma_tr_mm=0.01, sapma_mm_kare=0.002, tohum=0):
    """Gercek pozlara rastgele yuruyus (random walk) + sabit egilim (bias) seklinde biriken IMU kaymasi ekler.

    Her gecis basinda kayma sifirlanmaz; tarama boyunca birikir (kotu senaryo).
    """
    rng = np.random.RandomState(tohum)
    kayma = np.eye(4)
    bias = rng.normal(size=3); bias = bias / np.linalg.norm(bias) * sapma_mm_kare
    for k in kayitlar:
        d = np.eye(4)
        d[:3, :3] = _rot(rng, sigma_rot_derece)
        d[:3, 3] = rng.normal(0, sigma_tr_mm, 3) + bias
        kayma = kayma @ d
        k["T_imu"] = k["T"] @ kayma
    return kayitlar


def birlestir(kayitlar, mod="ideal", yama_kare=40, voxel=0.25, kapi_mm=3.0, kapi_derece=5.0):
    """Noktalari dunya koordinatina tasir.

    ideal: gercek pozlar.
    imu_ham: kaymali IMU pozlari, hicbir duzeltme yok (karsilastirma).
    imu:   kaymali IMU pozlari; kareler 'yama_kare'lik yamalar halinde toplanir ve her yama mevcut
           modele hibrit ICP ile hizalanir. Elde edilen duzeltme sonraki yamalara aktarilir
           (IMU kisa vadeli hareketi, ICP uzun vadeli kaymayi duzeltir). IMU hatasi sinirli
           oldugu icin 'kapi_mm'/'kapi_derece'den buyuk duzeltmeler reddedilir (makullük kapisi).
    """
    def dunya(k, T):
        return k["P_kam"] @ T[:3, :3].T + T[:3, 3]

    def merkezler():
        return np.concatenate([np.repeat(k["T_tahmin"][:3, 3][None], len(k["P_kam"]), 0) for k in kayitlar])

    if mod == "ideal":
        for k in kayitlar:
            k["T_tahmin"] = k["T"]
        return np.concatenate([dunya(k, k["T"]) for k in kayitlar]), {"kamera": merkezler()}
    if mod == "imu_ham":
        for k in kayitlar:
            k["T_tahmin"] = k["T_imu"]
        return np.concatenate([dunya(k, k["T_imu"]) for k in kayitlar]), {"kamera": merkezler()}

    st = PointCloudStitcher(voxel_size=voxel * 1.6, registration_mode="hybrid", seed=0)
    duzeltme = np.eye(4)
    model = None
    log = []
    for bas in range(0, len(kayitlar), yama_kare):
        yama = kayitlar[bas:bas + yama_kare]
        P = np.concatenate([dunya(k, duzeltme @ k["T_imu"]) for k in yama if len(k["P_kam"])] or [np.empty((0, 3))])
        if len(P) < 200:
            for k in yama:
                k["T_tahmin"] = duzeltme @ k["T_imu"]
            continue
        p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
        if model is None:
            D = np.eye(4)
            model = p.voxel_down_sample(voxel)
        else:
            D = st.register(p, model, max_correction_mm=kapi_mm, max_correction_deg=kapi_derece)
            log.append(dict(yama=bas // yama_kare, gecis=yama[0]["gecis"], **st.last_result))
            if not st.last_result.get("valid", False):
                D = np.eye(4)     # guvenilmez hizalama: IMU tahminini koru
            model += p.transform(D)
            model = model.voxel_down_sample(voxel)
        duzeltme = D @ duzeltme
        for k in yama:
            k["T_tahmin"] = duzeltme @ k["T_imu"]
    return np.concatenate([dunya(k, k["T_tahmin"]) for k in kayitlar]), {"hizalamalar": log, "kamera": merkezler()}


def poz_hatasi(kayitlar):
    """Her karenin tahmini pozundan dogan nokta sapmasi (mm, karenin kendi noktalarinda)."""
    h = []
    for k in kayitlar:
        if len(k["P_kam"]):
            a = k["P_kam"] @ k["T_tahmin"][:3, :3].T + k["T_tahmin"][:3, 3]
            b = k["P_kam"] @ k["T"][:3, :3].T + k["T"][:3, 3]
            h.append(np.sqrt(np.mean(np.sum((a - b) ** 2, 1))))
    return np.array(h)


# ───────────────────────── 3. mesh + degerlendirme ─────────────────────────

def mesh_olustur(P, kamera_merkezleri, derinlik=9, voxel=0.1, kirpma_mm=0.4):
    """Nokta bulutundan Poisson yuzeyi.

    Normaller, her noktanin goruldugu kamera konumuna dogru yonlendirilir (tarayicilarin standart
    yontemi; orient_normals_consistent_tangent_plane'e gore hem dogru hem cok daha hafif).
    """
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pcd.normals = o3d.utility.Vector3dVector(kamera_merkezleri - P)      # gecici: bakis yonu
    pcd = pcd.voxel_down_sample(voxel)                                    # normaller ortalanir
    bakis = np.asarray(pcd.normals).copy()
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)
    pts = np.asarray(pcd.points)
    _, ix = en_yakin(np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)).voxel_down_sample(voxel).points), pts)
    bakis = bakis[ix]
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.6, max_nn=30))
    n = np.asarray(pcd.normals)
    ters = np.sum(n * bakis, axis=1) < 0
    n[ters] *= -1
    pcd.normals = o3d.utility.Vector3dVector(n)

    mesh, yogunluk = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=derinlik)
    yogunluk = np.asarray(yogunluk)
    mesh.remove_vertices_by_mask(yogunluk < np.percentile(yogunluk, 2))
    # Taranmamis bolgelerde Poisson'un uydurdugu yuzeyleri kirp (noktalara uzak koseler)
    d, _ = en_yakin(pts, np.asarray(mesh.vertices))
    mesh.remove_vertices_by_mask(d > kirpma_mm)
    mesh = MeshGenerator(method="poisson", smooth_iterations=1).smooth_and_clean_mesh(mesh)
    return mesh, pcd


def gt_sahnesi(v, f):
    m = o3d.t.geometry.TriangleMesh()
    m.vertex.positions = o3d.core.Tensor(v.astype(np.float32))
    m.triangle.indices = o3d.core.Tensor(f.astype(np.int32))
    s = o3d.t.geometry.RaycastingScene()
    s.add_triangles(m)
    return s


def uzaklik(sahne, P):
    return sahne.compute_distance(o3d.core.Tensor(np.asarray(P, dtype=np.float32))).numpy()


def en_iyi_hizalama(P, v, f):
    """Taramanin referansa en iyi rijit hizalanmasi (ISO 12836 'trueness' olcumunden once yapilir)."""
    gt = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v.astype(np.float64)), o3d.utility.Vector3iVector(f.astype(np.int32)))
    gt.compute_vertex_normals()
    hedef = gt.sample_points_uniformly(300000)
    kaynak = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)).voxel_down_sample(0.2)
    T = np.eye(4)
    for esik in (2.0, 0.5, 0.2):
        T = o3d.pipelines.registration.registration_icp(
            kaynak, hedef, esik, T, o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60)).transformation
    return T


def degerlendir(v, f, etiket, P, mesh, hizala=False):
    sahne = gt_sahnesi(v, f)
    if hizala:
        T = en_iyi_hizalama(P, v, f)
        P = P @ T[:3, :3].T + T[:3, 3]
        mesh = o3d.geometry.TriangleMesh(mesh).transform(T)
    d_nokta = uzaklik(sahne, P)
    Vr = np.asarray(mesh.vertices)
    d_mesh = uzaklik(sahne, Vr)
    rs = o3d.t.geometry.RaycastingScene()
    rs.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    d_kapsama = rs.compute_distance(o3d.core.Tensor(v.astype(np.float32))).numpy()
    dis = etiket > 0
    ozet = lambda d: dict(ortalama=float(d.mean()), medyan=float(np.median(d)), rms=float(np.sqrt((d ** 2).mean())),
                          p95=float(np.percentile(d, 95)), maks=float(d.max()))
    return {
        "nokta_sayisi": int(len(P)),
        "nokta_dogrulugu_mm": ozet(d_nokta),
        "mesh_kose_sayisi": int(len(Vr)),
        "mesh_dogrulugu_mm": ozet(d_mesh),
        "kapsama_dis_0.1mm": float((d_kapsama[dis] < 0.1).mean()),
        "kapsama_dis_0.2mm": float((d_kapsama[dis] < 0.2).mean()),
        "kapsama_dis_eti_0.2mm": float((d_kapsama[~dis] < 0.2).mean()),
    }, d_mesh, d_kapsama


def segmentasyon(mesh, v, etiket, checkpoint, cikti, tur=2):
    import torch
    from ai_model.network import checkpoint_yukle
    from ai_model.inference import koseleri_tahmin_et
    from ai_model.evaluate import karisiklik_matrisi, matristen_metrikler
    model, kayit = checkpoint_yukle(checkpoint, "cpu")
    nokta = int(kayit.get("num_points", 8192))
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    Vr = np.asarray(mesh.vertices, dtype=np.float32)
    Nr = kose_normalleri(Vr, np.asarray(mesh.triangles))
    t0 = time.time()
    tahmin, _ = koseleri_tahmin_et(model, Vr, Nr, nokta_sayisi=nokta, cihaz="cpu", tur_sayisi=tur, batch_boyutu=1)
    # Gercek etiketi en yakin GT kosesinden tasi
    _, idx = en_yakin(v, Vr)
    hedef = etiket[idx]
    m = matristen_metrikler(karisiklik_matrisi(tahmin, hedef))
    kayitci.info(f"Segmentasyon (yeniden olusturulan mesh): mIoU(dis) {m['mIoU_teeth']:.3f}, {time.time() - t0:.0f} sn")
    return tahmin, hedef, {"mIoU_dis": m["mIoU_teeth"], "dogruluk": m["accuracy"], "dice": m["dice"],
                           "model": kayit.get("model_name"), "kose": int(len(Vr))}


# ───────────────────────── ana akis ─────────────────────────

def calistir(mesh_yolu, cikti_kok, mod="ideal", yeniden_tara=False, goruntu_gurultusu=3.0, checkpoint=None,
             imu=None, kalibrasyon_hatasi=None, segmentle=True, etiket_eki="", kare_dosyasi="kareler.npz"):
    v, f = obj_oku(mesh_yolu)
    _, _, etiket, hasta = mesh_oku(mesh_yolu, os.path.splitext(mesh_yolu)[0] + ".json")
    ad = os.path.splitext(os.path.basename(mesh_yolu))[0]
    cikti = os.path.join(cikti_kok, ad)
    os.makedirs(cikti, exist_ok=True)

    kare_dosya = os.path.join(cikti, kare_dosyasi)
    if yeniden_tara or not os.path.exists(kare_dosya):
        tara(v, f, etiket, cikti, goruntu_gurultusu=goruntu_gurultusu, dosya=kare_dosyasi)
    kayitlar, K, lazer = kareleri_yukle(cikti, kare_dosyasi)

    # Kalibrasyon hatasi senaryosu: goruntu dogru K ile uretildi, hat hatali K / lazer duzlemiyle calisir
    if kalibrasyon_hatasi:
        K = K.copy(); lazer = lazer.copy()
        K[0, 0] *= 1 + kalibrasyon_hatasi.get("odak_orani", 0); K[1, 1] *= 1 + kalibrasyon_hatasi.get("odak_orani", 0)
        a = np.arctan2(lazer[2], lazer[1]) + np.radians(kalibrasyon_hatasi.get("lazer_aci_derece", 0))
        b = -lazer[3] / lazer[1] + kalibrasyon_hatasi.get("baz_mm", 0)
        lazer = np.array([0.0, np.cos(a), np.sin(a), -b * np.cos(a)])
    kayitlar = uc_boyuta(kayitlar, K, lazer)
    if mod in ("imu", "imu_ham"):
        kayitlar = imu_pozlari(kayitlar, **(imu or {}))
    P, bilgi = birlestir(kayitlar, mod)
    ph = poz_hatasi(kayitlar)

    mesh, pcd = mesh_olustur(P, bilgi["kamera"])
    sonuc, d_mesh, d_kapsama = degerlendir(v, f, etiket, P, mesh)
    if mod != "ideal":
        hz, d_mesh_hz, d_kap_hz = degerlendir(v, f, etiket, P, mesh, hizala=True)
        sonuc["en_iyi_hizalama_sonrasi"] = {k: hz[k] for k in ("nokta_dogrulugu_mm", "mesh_dogrulugu_mm",
                                                               "kapsama_dis_0.1mm", "kapsama_dis_0.2mm")}
        np.save(os.path.join(cikti, f"sapma_mesh_{mod}{etiket_eki}_hizali.npy"), d_mesh_hz.astype(np.float32))
    sonuc.update(dict(hasta=hasta, mod=mod, kare=len(kayitlar), goruntu_gurultusu=goruntu_gurultusu,
                      poz_hatasi_ort_mm=float(ph.mean()), poz_hatasi_maks_mm=float(ph.max()),
                      kalibrasyon_hatasi=kalibrasyon_hatasi or {}, imu=imu or {}))
    if mod in ("imu", "imu_ham"):
        sonuc["imu_ham_poz_hatasi_ort_mm"] = float(np.mean([
            np.sqrt(np.mean(np.sum(((k["P_kam"] @ k["T_imu"][:3, :3].T + k["T_imu"][:3, 3]) -
                                    (k["P_kam"] @ k["T"][:3, :3].T + k["T"][:3, 3])) ** 2, 1)))
            for k in kayitlar if len(k["P_kam"])]))
        hz = bilgi.get("hizalamalar") or []
        sonuc["yama_hizalama_gecerli_orani"] = float(np.mean([h["valid"] for h in hz])) if hz else None
        sonuc["yama_hizalamalari"] = hz
        sonuc["gecis_basina_poz_hatasi_mm"] = {g: float(np.mean([e for k, e in zip([k for k in kayitlar if len(k["P_kam"])], ph) if k["gecis"] == gi]))
                                               for gi, (g, _) in enumerate(GECISLER)}

    etiket_ad = f"{mod}{etiket_eki}"
    o3d.io.write_triangle_mesh(os.path.join(cikti, f"rekonstruksiyon_{etiket_ad}.ply"), mesh)
    np.save(os.path.join(cikti, f"sapma_mesh_{etiket_ad}.npy"), d_mesh.astype(np.float32))
    np.save(os.path.join(cikti, f"kapsama_{etiket_ad}.npy"), d_kapsama.astype(np.float32))

    if segmentle and checkpoint and os.path.exists(checkpoint):
        tahmin, hedef, seg = segmentasyon(mesh, v, etiket, checkpoint, cikti)
        np.save(os.path.join(cikti, f"seg_tahmin_{etiket_ad}.npy"), tahmin.astype(np.int8))
        np.save(os.path.join(cikti, f"seg_hedef_{etiket_ad}.npy"), hedef.astype(np.int8))
        sonuc["segmentasyon"] = seg

    with open(os.path.join(cikti, f"sonuc_{etiket_ad}.json"), "w", encoding="utf-8") as fh:
        json.dump(sonuc, fh, indent=2, ensure_ascii=False)
    kayitci.info(f"[{ad} | {etiket_ad}] nokta dogrulugu medyan {sonuc['nokta_dogrulugu_mm']['medyan']:.4f} mm, "
                 f"mesh RMS {sonuc['mesh_dogrulugu_mm']['rms']:.4f} mm, kapsama(dis,0.2mm) %{sonuc['kapsama_dis_0.2mm'] * 100:.1f}")
    return sonuc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Uctan uca sanal tarama deneyi")
    p.add_argument("--mesh", required=True)
    p.add_argument("--out", default="runs/sanal_tarama")
    p.add_argument("--mode", choices=["ideal", "imu", "imu_ham"], default="ideal")
    p.add_argument("--checkpoint", default="ai_model/best_dental_model.pth")
    p.add_argument("--rescan", action="store_true")
    p.add_argument("--noise", type=float, default=3.0, help="Goruntu gurultusu (0-255 olcegi)")
    p.add_argument("--no-seg", action="store_true")
    a = p.parse_args()
    s = calistir(a.mesh, a.out, a.mode, a.rescan, a.noise, a.checkpoint, segmentle=not a.no_seg)
    print(json.dumps(s, indent=2, ensure_ascii=False))
