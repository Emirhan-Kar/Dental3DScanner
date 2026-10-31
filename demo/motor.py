"""
Demo motoru: sanal taramayi arka planda calistirir ve arayuzun okuyacagi durumu gunceller.

Asamalar: hazirlik -> tarama (goruntu + lazer tespiti + ucgenleme) -> birlestirme -> yuzey
          -> degerlendirme -> segmentasyon -> bitti
Her tamamlanan tarama 'runs/demo/<anahtar>/' altina kaydedilir; ayni ayarlarla tekrar
istendiginde kayittan hizlica oynatilir (sunumda bekleme olmamasi icin).
"""
import os
import sys
import json
import time
import glob
import hashlib
import threading
import numpy as np
import open3d as o3d
import cv2

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from ai_model.dataset import obj_oku, mesh_oku, kose_normalleri, FDI_SINIFLARI, mesh_geometri_oku
from simulation.sanal_tarayici import SanalTarayici, ark_egrisi, tarama_pozlari
from processing.laser_detector import LaserDetector
from processing.triangulation import Triangulator
from experiments import sanal_tarama_deneyi as sd
from ui.viewer import get_fdi_color_palette
from processing import klinik_olcum as ko

ASAMALAR = [
    ("hazirlik", "Hazırlık"),
    ("tarama", "Görüntü + lazer tespiti + üçgenleme"),
    ("birlestirme", "Birleştirme"),
    ("yuzey", "Yüzey (Poisson)"),
    ("degerlendirme", "Doğruluk ölçümü"),
    ("segmentasyon", "Diş segmentasyonu (DGCNN)"),
    ("bitti", "Tamamlandı"),
]
GECIS_ACILARI = {"tepe": 0.0, "yanak": 35.0, "dil": -35.0}
PALET = get_fdi_color_palette()
GORUNTU_KARE_SAKLA = 3          # kayit icin her n. kareyi JPEG olarak sakla
GOSTERIM_UCGEN = 70000          # tarayicida gosterilecek mesh boyutu
GOSTERIM_NOKTA = 60000


def cihaz():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def vakalari_listele():
    """data/3DTeethData altindaki etiketli cene modelleri; test setinde olanlar isaretlenir."""
    test = set()
    sp = os.path.join(KOK, "runs", "dgcnn", "split.json")
    if os.path.exists(sp):
        try:
            test = {os.path.basename(o) for o, _ in json.load(open(sp, encoding="utf-8"))["test"]}
        except Exception:
            pass
    vakalar = []
    for obj in sorted(glob.glob(os.path.join(KOK, "data", "3DTeethData", "*", "*", "*", "*.obj"))):
        if os.path.exists(os.path.splitext(obj)[0] + ".json"):
            vakalar.append({"yol": obj, "ad": os.path.splitext(os.path.basename(obj))[0],
                            "test": os.path.basename(obj) in test})
    vakalar.sort(key=lambda v: (not v["test"], v["ad"]))
    return vakalar


def _jpeg(img, kalite=80):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, kalite])
    return buf.tobytes() if ok else b""


def gosterim_icin_parlat(img):
    """Sadece ekranda gostermek icin: gamma + kontrast. Tespit algoritmasi ham goruntuyu kullanir."""
    f = img.astype(np.float32) / 255.0
    f = np.power(f, 0.55) * 1.15
    return np.clip(f * 255, 0, 255).astype(np.uint8)


def tespit_gorseli(img, tepeler):
    """Acik gri zemin uzerinde tespit edilen lazer merkezleri (kalin yesil)."""
    g = cv2.cvtColor(gosterim_icin_parlat(img), cv2.COLOR_BGR2GRAY)
    g = cv2.addWeighted(g, 0.55, np.full_like(g, 255), 0.45, 0)
    vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    for u, v in tepeler:
        cv2.circle(vis, (int(round(u)), int(round(v))), 2, (40, 170, 40), -1, lineType=cv2.LINE_AA)
    return vis


class MiniHarita:
    """Cenenin ustten gorunumu: probun konumu, anlik lazer cizgisi ve taranan bolge."""

    def __init__(self, v, etiket, yukari, boyut=480):
        self.boyut = boyut
        xy = v[:, :2]
        self.min = xy.min(0) - 4.0
        aralik = (xy.max(0) + 4.0 - self.min).max()
        self.olcek = (boyut - 1) / aralik
        self.ofs = (boyut - 1 - (xy.max(0) + 4.0 - self.min) * self.olcek) / 2
        sira = np.argsort(v @ yukari)                 # ustteki yuzeyler en son cizilsin
        uv = self._piksel(xy[sira])
        taban = np.full((boyut, boyut, 3), 255, np.uint8)
        renk = np.where((etiket[sira] > 0)[:, None], [[205, 205, 200]], [[190, 170, 225]]).astype(np.uint8)
        taban[uv[:, 1], uv[:, 0]] = renk
        taban = cv2.erode(taban, np.ones((3, 3), np.uint8))          # bosluklari kapat (beyaz zeminde min)
        self.taban = taban
        self.taranan = np.zeros((boyut, boyut), np.uint8)

    def _piksel(self, xy):
        p = (xy - self.min) * self.olcek + self.ofs
        p[:, 1] = self.boyut - 1 - p[:, 1]           # y yukari
        return np.clip(np.round(p).astype(int), 0, self.boyut - 1)

    def ciz(self, P_dunya, kamera_merkezi):
        if len(P_dunya):
            uv = self._piksel(P_dunya[:, :2])
            self.taranan[uv[:, 1], uv[:, 0]] = 255
        img = self.taban.copy()
        maske = cv2.dilate(self.taranan, np.ones((3, 3), np.uint8)) > 0
        img[maske] = (0.45 * img[maske] + 0.55 * np.array([170, 130, 40])).astype(np.uint8)   # taranan: celik mavi ton
        if len(P_dunya):
            for a, b in zip(uv[:-1], uv[1:]):
                if abs(a - b).sum() < 12:
                    cv2.line(img, tuple(int(x) for x in a), tuple(int(x) for x in b), (40, 50, 215), 3, cv2.LINE_AA)
        c = self._piksel(np.asarray(kamera_merkezi)[None, :2])[0]
        cv2.circle(img, (int(c[0]), int(c[1])), 9, (48, 36, 27), 2, cv2.LINE_AA)
        cv2.circle(img, (int(c[0]), int(c[1])), 3, (48, 36, 27), -1, cv2.LINE_AA)
        return img


KARE_BASLIK = b"DSKARE01"     # dosya JPEG imzasiyla baslamasin: bazi aktarim araclari JPEG sanip icine veri ekliyor


def kareleri_kaydet(dizin, kareler):
    """(kare_no, jpeg_bayt) listesini tek dosyada saklar: kareler.bin + kareler_indeks.npy."""
    with open(os.path.join(dizin, "kareler.bin"), "wb") as fh:
        fh.write(KARE_BASLIK)
        for _, b in kareler:
            fh.write(b)
    indeks = np.array([[i, len(b)] for i, b in kareler], dtype=np.int64).reshape(-1, 2)
    np.save(os.path.join(dizin, "kareler_indeks.npy"), indeks)


def kareleri_oku(dizin):
    yol = os.path.join(dizin, "kareler.bin")
    if not os.path.exists(yol):
        return []
    indeks = np.load(os.path.join(dizin, "kareler_indeks.npy"))
    veri = open(yol, "rb").read()
    ofs = len(KARE_BASLIK) if veri.startswith(KARE_BASLIK) else 0
    # Eski bicim (basliksiz) dosyalarda aktarim sirasinda ilk JPEG'e ek veri (or. C2PA etiketi) eklenmis
    # olabilir: fazlalik ilk karenin uzunluguna eklenir, boylece sonraki kareler kaymaz.
    fazla = len(veri) - ofs - int(indeks[:, 1].sum()) if len(indeks) else 0
    sonuc = []
    for j, (i, n) in enumerate(indeks):
        n = int(n) + (fazla if j == 0 and fazla > 0 else 0)
        sonuc.append((int(i), veri[ofs:ofs + n]))
        ofs += n
    return sonuc


def _seyrelt_mesh(mesh, ucgen=GOSTERIM_UCGEN):
    if len(mesh.triangles) > ucgen:
        mesh = mesh.simplify_quadric_decimation(ucgen)
    mesh.remove_unreferenced_vertices()
    return mesh


class TaramaMotoru:
    def __init__(self):
        self.kilit = threading.Lock()
        self._is = None
        self._dur = threading.Event()
        self.sifirla()

    # ─────────── durum ───────────
    def sifirla(self):
        with getattr(self, "kilit", threading.Lock()):
            self.durum = {
                "asama": None, "calisiyor": False, "hata": None, "mesajlar": [],
                "kare": 0, "kare_toplam": 0, "gecis": "", "kayittan": False,
                "goruntu": None, "nokta": np.empty((0, 3), np.float32), "nokta_surum": 0,
                "sonuc": None, "yuzey": None, "seg": None, "baslangic": None, "sure": None, "olcum": None,
            }

    def oku(self, *anahtarlar):
        with self.kilit:
            return {k: self.durum.get(k) for k in anahtarlar} if anahtarlar else dict(self.durum)

    def _yaz(self, **kw):
        with self.kilit:
            self.durum.update(kw)

    def _mesaj(self, m):
        with self.kilit:
            self.durum["mesajlar"] = (self.durum["mesajlar"] + [f"{time.strftime('%H:%M:%S')}  {m}"])[-60:]

    @property
    def calisiyor(self):
        return self._is is not None and self._is.is_alive()

    def durdur(self):
        self._dur.set()

    # ─────────── baslat ───────────
    def baslat(self, vaka_yolu, gecisler=("tepe", "yanak", "dil"), adim_mm=0.5, mod="ideal",
               gurultu=3.0, kayittan_izin=True):
        if self.calisiyor:
            return False
        self.sifirla()
        self._dur.clear()
        ayar = dict(vaka=os.path.basename(vaka_yolu), gecisler=list(gecisler), adim=float(adim_mm),
                    mod=mod, gurultu=float(gurultu))
        anahtar = hashlib.md5(json.dumps(ayar, sort_keys=True).encode()).hexdigest()[:10]
        kayit_dizin = os.path.join(KOK, "runs", "demo", f"{ayar['vaka']}_{anahtar}")
        hedef = self._oynat if (kayittan_izin and os.path.exists(os.path.join(kayit_dizin, "sonuc.json"))) else self._calistir
        self._is = threading.Thread(target=self._guvenli, args=(hedef, vaka_yolu, ayar, kayit_dizin), daemon=True)
        self._is.start()
        return True

    def _guvenli(self, hedef, *a):
        self._yaz(calisiyor=True, baslangic=time.time())
        try:
            hedef(*a)
        except InterruptedError:
            self._mesaj("Tarama kullanıcı tarafından durduruldu.")
            self._yaz(asama=None)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._yaz(hata=f"{type(e).__name__}: {e}")
            self._mesaj(f"HATA: {e}")
        finally:
            b = self.oku("baslangic")["baslangic"]
            self._yaz(calisiyor=False, sure=time.time() - b if b else None)

    def _kontrol(self):
        if self._dur.is_set():
            raise InterruptedError()

    # ─────────── canli calistirma ───────────
    def _calistir(self, vaka_yolu, ayar, kayit_dizin):
        self._yaz(asama="hazirlik")
        self._mesaj(f"Model yükleniyor: {ayar['vaka']}")
        v, f = obj_oku(vaka_yolu)
        _, _, etiket, _ = mesh_oku(vaka_yolu, os.path.splitext(vaka_yolu)[0] + ".json")
        alt = v[etiket > 0, 2].mean() > v[etiket == 0, 2].mean()
        yukari = np.array([0, 0, 1.0]) if alt else np.array([0, 0, -1.0])
        st = SanalTarayici(v, f, etiket, goruntu_gurultusu=ayar["gurultu"], tohum=0)
        kam = st.kamera
        det = LaserDetector()
        tri = Triangulator(camera_matrix=kam.K, laser_plane=kam.lazer_duzlemi)
        ark, teget = ark_egrisi(v, etiket, FDI_SINIFLARI, ayar["adim"], yukari=yukari)
        harita = MiniHarita(v, etiket, yukari, boyut=kam.h)

        kayitlar = []
        for g, ad in enumerate(ayar["gecisler"]):
            for T in tarama_pozlari(ark, teget, yukari, egim_derece=GECIS_ACILARI[ad]):
                kayitlar.append(dict(gecis=g, gecis_ad=ad, T=T))
        if ayar["mod"] in ("imu", "imu_ham"):
            kayitlar = sd.imu_pozlari(kayitlar)
            self._mesaj("IMU kayması simüle ediliyor (rastgele yürüyüş + 2 µm/kare eğilim).")
        self._mesaj(f"{len(kayitlar)} kare taranacak ({', '.join(ayar['gecisler'])}; {ayar['adim']} mm aralık).")
        os.makedirs(kayit_dizin, exist_ok=True)
        saklanan = []                      # (kare_no, jpeg) — sonunda tek dosyaya yazilir

        # 1) Tarama
        self._yaz(asama="tarama", kare_toplam=len(kayitlar))
        canli, surum = [], 0
        for i, k in enumerate(kayitlar):
            self._kontrol()
            img, _ = st.kare_uret(k["T"])
            tepeler, _ = det.detect_subpixel_peaks(img)
            k["tepeler"] = np.array(tepeler, np.float32).reshape(-1, 2)
            k["P_kam"] = tri.pixels_to_3d(k["tepeler"]).astype(np.float64) if len(tepeler) else np.empty((0, 3))
            T_gor = k.get("T_imu", k["T"])           # canli gorunum: sistemin "bildigi" poz
            P_d = (k["P_kam"] @ T_gor[:3, :3].T + T_gor[:3, 3]) if len(k["P_kam"]) else np.empty((0, 3))
            if len(P_d):
                canli.append(P_d[::3].astype(np.float32))
            jpg = _jpeg(np.hstack([gosterim_icin_parlat(img), tespit_gorseli(img, tepeler),
                                   harita.ciz(P_d, k["T"][:3, 3])]))
            if i % GORUNTU_KARE_SAKLA == 0:
                saklanan.append((i, jpg))
            if i % 4 == 0 or i == len(kayitlar) - 1:
                P = np.concatenate(canli) if canli else np.empty((0, 3), np.float32)
                if len(P) > GOSTERIM_NOKTA:
                    P = P[np.linspace(0, len(P) - 1, GOSTERIM_NOKTA).astype(int)]
                surum += 1
                self._yaz(nokta=P, nokta_surum=surum)
            self._yaz(kare=i + 1, gecis=k["gecis_ad"], goruntu=jpg)
        kareleri_kaydet(kayit_dizin, saklanan)
        self._mesaj(f"Tarama bitti: {sum(len(k['P_kam']) for k in kayitlar) / 1e3:.0f} bin 3D nokta.")

        # 2) Birlestirme
        self._kontrol()
        self._yaz(asama="birlestirme")
        P, bilgi = sd.birlestir(kayitlar, ayar["mod"])
        ph = sd.poz_hatasi(kayitlar)
        if ayar["mod"] == "imu":
            hz = bilgi.get("hizalamalar", [])
            self._mesaj(f"Hibrit ICP: {sum(h['valid'] for h in hz)}/{len(hz)} yama hizalandı, "
                        f"{sum(h.get('rejected_by_gate', 0) for h in hz)} şüpheli düzeltme kapıda reddedildi.")
        if ayar["mod"] != "ideal":
            self._mesaj(f"Ortalama poz hatası: {ph.mean():.2f} mm")

        # 3) Yuzey
        self._kontrol()
        self._yaz(asama="yuzey")
        mesh, _ = sd.mesh_olustur(P, bilgi["kamera"])
        self._mesaj(f"Poisson yüzeyi: {len(mesh.vertices) / 1e3:.0f} bin köşe.")
        o3d.io.write_triangle_mesh(os.path.join(kayit_dizin, "yuzey.ply"), mesh)

        # 4) Degerlendirme
        self._kontrol()
        self._yaz(asama="degerlendirme")
        r, d_mesh, _ = sd.degerlendir(v, f, etiket, P, mesh)
        if ayar["mod"] != "ideal":
            r2, d_mesh, _ = sd.degerlendir(v, f, etiket, P, mesh, hizala=True)
            r.update({k: r2[k] for k in ("nokta_dogrulugu_mm", "mesh_dogrulugu_mm", "kapsama_dis_0.1mm", "kapsama_dis_0.2mm")})
        r["poz_hatasi_ort_mm"] = float(ph.mean())
        self._mesaj(f"Yüzey doğruluğu RMS {r['mesh_dogrulugu_mm']['rms']:.3f} mm, diş kapsaması %{r['kapsama_dis_0.2mm'] * 100:.1f}")

        # 5) Segmentasyon
        self._kontrol()
        self._yaz(asama="segmentasyon")
        tahmin, seg = self._segmentle(mesh, gercek=(v, etiket))
        r["segmentasyon"] = seg
        self._mesaj(f"Segmentasyon: {len(seg['disler'])} diş bulundu, mIoU {seg['mIoU_dis']:.3f} ({seg['cihaz']}, {seg['sure']:.0f} sn)")

        # Gosterim icin seyrelt + kaydet
        gos = _seyrelt_mesh(o3d.geometry.TriangleMesh(mesh))
        Vg = np.asarray(gos.vertices)
        dist, ix = sd.en_yakin(np.asarray(mesh.vertices), Vg)
        np.savez_compressed(os.path.join(kayit_dizin, "gosterim.npz"), V=Vg.astype(np.float32),
                            F=np.asarray(gos.triangles).astype(np.int32), sapma=d_mesh[ix].astype(np.float32),
                            etiket=tahmin[ix].astype(np.int8), nokta=self.oku("nokta")["nokta"])
        r.update(ayar=ayar, kare=len(kayitlar))
        with open(os.path.join(kayit_dizin, "sonuc.json"), "w", encoding="utf-8") as fh:
            json.dump(r, fh, indent=1, ensure_ascii=False)
        self._sonuclari_yayinla(kayit_dizin, r)

    def _sonuclari_yayinla(self, kayit_dizin, r):
        g = np.load(os.path.join(kayit_dizin, "gosterim.npz"))
        try:
            olcum = klinik_olcumler(kayit_dizin, g, r.get("ayar", {}).get("vaka"))
            if olcum["tahmin"]["ark_farki"] is not None:
                self._mesaj(f"Klinik ölçüm: ark uzunluğu farkı {olcum['tahmin']['ark_farki']:+.1f} mm "
                            f"({capraskilik_yorumu(olcum['tahmin']['ark_farki'])})")
        except Exception as e:
            olcum = None
            self._mesaj(f"Klinik ölçüm yapılamadı: {e}")
        self._yaz(sonuc=r, yuzey=dict(V=g["V"], F=g["F"], sapma=g["sapma"]),
                  seg=dict(V=g["V"], F=g["F"], etiket=g["etiket"]), olcum=olcum, asama="bitti")

    # ─────────── kayittan oynatma ───────────
    def _oynat(self, vaka_yolu, ayar, kayit_dizin):
        self._yaz(asama="hazirlik", kayittan=True)
        self._mesaj("Bu ayarlarla yapılmış kayıtlı tarama bulundu — hızlı oynatılıyor.")
        r = json.load(open(os.path.join(kayit_dizin, "sonuc.json"), encoding="utf-8"))
        g = np.load(os.path.join(kayit_dizin, "gosterim.npz"))
        kareler = kareleri_oku(kayit_dizin)
        P = g["nokta"]
        self._yaz(asama="tarama", kare_toplam=r["kare"])
        n = max(1, len(kareler))
        for j, (i, jpg) in enumerate(kareler):
            self._kontrol()
            m = int(len(P) * (j + 1) / n)
            self._yaz(goruntu=jpg, kare=i + 1, nokta=P[:m], nokta_surum=j)
            time.sleep(0.03)
        self._yaz(kare=r["kare"], nokta=P, nokta_surum=n + 1)
        for a in ("birlestirme", "yuzey", "degerlendirme", "segmentasyon"):
            self._kontrol()
            self._yaz(asama=a)
            time.sleep(0.4)
        self._mesaj(f"Yüzey doğruluğu RMS {r['mesh_dogrulugu_mm']['rms']:.3f} mm, diş kapsaması %{r['kapsama_dis_0.2mm'] * 100:.1f}")
        self._mesaj(f"Segmentasyon: {len(r['segmentasyon']['disler'])} diş, mIoU {r['segmentasyon']['mIoU_dis']:.3f}")
        self._sonuclari_yayinla(kayit_dizin, r)

    # ─────────── segmentasyon ───────────
    def _segmentle(self, mesh, gercek=None, tur=1):
        import torch
        from ai_model.network import checkpoint_yukle
        from ai_model.inference import koseleri_tahmin_et
        from ai_model.evaluate import karisiklik_matrisi, matristen_metrikler
        c = cihaz()
        model, kayit = checkpoint_yukle(os.path.join(KOK, "ai_model", "best_dental_model.pth"), c)
        Vr = np.asarray(mesh.vertices, dtype=np.float32)
        Nr = kose_normalleri(Vr, np.asarray(mesh.triangles))
        t0 = time.time()
        tahmin, _ = koseleri_tahmin_et(model, Vr, Nr, nokta_sayisi=int(kayit.get("num_points", 8192)), cihaz=c,
                                       tur_sayisi=tur, batch_boyutu=8 if c == "cuda" else 1)
        disler = sorted(int(FDI_SINIFLARI[i]) for i in np.unique(tahmin) if i > 0 and (tahmin == i).sum() > 200)
        seg = {"disler": disler, "cihaz": c, "sure": time.time() - t0, "model": kayit.get("model_name")}
        if gercek is not None:
            v, etiket = gercek
            _, idx = sd.en_yakin(v, Vr)
            m = matristen_metrikler(karisiklik_matrisi(tahmin, etiket[idx]))
            seg.update(mIoU_dis=float(m["mIoU_teeth"]), dogruluk=float(m["accuracy"]),
                       dis_iou={int(FDI_SINIFLARI[i]): float(m["iou_per_class"][i]) for i in range(1, 33)
                                if not np.isnan(m["iou_per_class"][i])})
        return tahmin, seg

    # ─────────── dosya yukleyip segmentleme ───────────
    def dosya_segmentle(self, yol):
        v, f = mesh_geometri_oku(yol)
        if len(v) < 100 or len(f) == 0:
            raise ValueError("Dosyada yeterli üçgen yüzey bulunamadı (nokta bulutu değil, mesh gerekli).")
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v.astype(np.float64)),
                                         o3d.utility.Vector3iVector(f.astype(np.int32)))
        tahmin, seg = self._segmentle(mesh)
        gos = _seyrelt_mesh(o3d.geometry.TriangleMesh(mesh))
        Vg = np.asarray(gos.vertices)
        _, ix = sd.en_yakin(v, Vg)
        seg["olcum"] = olcum_paketi(np.asarray(mesh.vertices), np.asarray(mesh.triangles), tahmin)
        return dict(V=Vg.astype(np.float32), F=np.asarray(gos.triangles), etiket=tahmin[ix]), seg


# ─────────────────────────── iki cene birlikte (kapanis) ───────────────────────────
#
# Teeth3DS her ceneyi ayri ve "disler +z yonune bakacak" sekilde kaydeder; hastanin isirma kaydi
# (iki cenenin birbirine gore gercek konumu) veri setinde yoktur. Bu yuzden ust cene y ekseni
# etrafinda 180° dondurulur (disler asagi bakar, hastanin sagi alt ceneyle ayni tarafa gelir) ve
# dis tepeleri alt cenenin tepelerine oturacak sekilde yerlestirilir. Sonuc yaklasik bir kapanistir.

KAPANIS_DIZIN = os.path.join(KOK, "runs", "demo", "kapanis")
UST_DONME = np.diag([-1.0, 1.0, -1.0])           # y ekseni etrafinda 180° (det = +1, ucgen yonu korunur)


def hasta_ciftleri():
    """Hem alt hem ust cenesi bulunan hastalar: {hasta: {"lower": yol, "upper": yol, "test": bool}}."""
    ciftler = {}
    for v in vakalari_listele():
        hasta, _, cene = v["ad"].rpartition("_")
        if cene in ("lower", "upper"):
            c = ciftler.setdefault(hasta, {"test": False})
            c.setdefault(cene, v["yol"])
            c["test"] = c["test"] or v["test"]
    return {h: c for h, c in ciftler.items() if "lower" in c and "upper" in c}


def _ayar_etiketi(ayar):
    mod = {"ideal": "konum biliniyor", "imu_ham": "sadece IMU", "imu": "IMU + ICP"}.get(ayar.get("mod"), ayar.get("mod"))
    return f"{str(ayar.get('adim')).replace('.', ',')} mm · {mod} · {len(ayar.get('gecisler', []))} geçiş"


def kapanis_taramalari(hasta):
    """Bu hastanin iki cenesi icin AYNI ayarlarla yapilmis kayitli taramalar."""
    bulunan = {}
    for d in glob.glob(os.path.join(KOK, "runs", "demo", f"{hasta}_*")):
        sj = os.path.join(d, "sonuc.json")
        if not (os.path.exists(sj) and os.path.exists(os.path.join(d, "gosterim.npz"))):
            continue
        try:
            ayar = json.load(open(sj, encoding="utf-8"))["ayar"]
        except Exception:
            continue
        cene = os.path.splitext(ayar["vaka"])[0].rpartition("_")[2]
        geri = {k: v for k, v in ayar.items() if k != "vaka"}
        bulunan.setdefault(json.dumps(geri, sort_keys=True), {})[cene] = d
    sira = {"ideal": 0, "imu": 1, "imu_ham": 2}
    sonuc = [dict(ayar=json.loads(a), lower=c["lower"], upper=c["upper"]) for a, c in bulunan.items()
             if "lower" in c and "upper" in c]
    sonuc.sort(key=lambda s: (s["ayar"]["adim"], sira.get(s["ayar"]["mod"], 9)))
    for s in sonuc:
        s["etiket"] = _ayar_etiketi(s["ayar"])
    return sonuc


def tarama_gosterimi(dizin):
    """Kayitli taramanin gosterim mesh'i + ozet sonuclari."""
    g = np.load(os.path.join(dizin, "gosterim.npz"))
    r = json.load(open(os.path.join(dizin, "sonuc.json"), encoding="utf-8"))
    s = r["segmentasyon"]
    return dict(V=g["V"].astype(np.float32), F=g["F"].astype(np.int32), etiket=g["etiket"].astype(int),
                sapma=g["sapma"].astype(np.float32), disler=s["disler"],
                dis_iou={int(k): v for k, v in s.get("dis_iou", {}).items()},
                mIoU=s.get("mIoU_dis"), rms=r["mesh_dogrulugu_mm"]["rms"])


def orijinal_gosterimi(yol, motor=None):
    """Veri setindeki orijinal model + DGCNN segmentasyonu (bir kez hesaplanir, sonra kayittan okunur)."""
    os.makedirs(KAPANIS_DIZIN, exist_ok=True)
    ad = os.path.splitext(os.path.basename(yol))[0]
    npz, js = os.path.join(KAPANIS_DIZIN, ad + ".npz"), os.path.join(KAPANIS_DIZIN, ad + ".json")
    if not (os.path.exists(npz) and os.path.exists(js)):
        v, f = obj_oku(yol)
        _, _, etiket, _ = mesh_oku(yol, os.path.splitext(yol)[0] + ".json")
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v.astype(np.float64)),
                                         o3d.utility.Vector3iVector(np.asarray(f, dtype=np.int32)))
        tahmin, seg = (motor or TaramaMotoru())._segmentle(mesh, gercek=(v, etiket))
        gos = _seyrelt_mesh(o3d.geometry.TriangleMesh(mesh), GOSTERIM_UCGEN // 2)
        Vg = np.asarray(gos.vertices)
        _, ix = sd.en_yakin(v, Vg)
        np.savez_compressed(npz, V=Vg.astype(np.float32), F=np.asarray(gos.triangles).astype(np.int32),
                            etiket=tahmin[ix].astype(np.int8))
        with open(js, "w", encoding="utf-8") as fh:
            json.dump(seg, fh, ensure_ascii=False)
    g = np.load(npz)
    seg = json.load(open(js, encoding="utf-8"))
    return dict(V=g["V"].astype(np.float32), F=g["F"].astype(np.int32), etiket=g["etiket"].astype(int), sapma=None,
                disler=seg["disler"], dis_iou={int(k): v for k, v in seg.get("dis_iou", {}).items()},
                mIoU=seg.get("mIoU_dis"), rms=None)


def kapanis_konumu(alt, ust, ortusme_mm=0.8):
    """Ust cene icin (R, t): x' = R x + t. Aciklik sonradan t[2]'ye eklenir.

    Disler (etiket > 0) uzerinden: xy'de dis arklarinin merkezleri cakistirilir, z'de ust cenenin en alt
    dis tepeleri alt cenenin en ust tepelerine 'ortusme_mm' kadar girecek sekilde oturtulur
    (tuberkuller birbirinin arasina girer)."""
    R = UST_DONME
    da = alt["V"][alt["etiket"] > 0] if (alt["etiket"] > 0).sum() > 100 else alt["V"]
    du = ust["V"][ust["etiket"] > 0] if (ust["etiket"] > 0).sum() > 100 else ust["V"]
    du = du @ R.T
    t = np.zeros(3)
    t[:2] = da[:, :2].mean(0) - du[:, :2].mean(0)
    t[2] = np.percentile(da[:, 2], 99.5) - np.percentile(du[:, 2], 0.5) - ortusme_mm
    return R, t


# ─────────────────────────── klinik olcumler ───────────────────────────

FDI_DIZI = np.array(FDI_SINIFLARI)


def capraskilik_yorumu(fark):
    if fark is None:
        return "hesaplanamadı"
    if fark <= -1:
        return "dişler arasında boşluk var"
    if fark <= 1:
        return "dişler arka sığıyor, belirgin çapraşıklık yok"
    if fark <= 4:
        return "hafif çapraşıklık"
    if fark <= 8:
        return "orta çapraşıklık"
    return "şiddetli çapraşıklık"


def olcum_paketi(V, F, etiket_idx, cene=None):
    """Olcum ozeti + 3D gosterim icin ark egrisi ve genislik cizgileri."""
    o = ko.olc(V, FDI_DIZI[np.asarray(etiket_idx).astype(int)], F, cene=cene)
    gorsel = dict(egri=None if not o["ark"] else o["ark"]["egri"].astype(np.float32),
                  cizgiler={f: (d["uc_sag_3d"], d["uc_sol_3d"]) for f, d in o["disler"].items()})
    return dict(ozet=ko.ozet(o), gorsel=gorsel, ham=o)


def _vaka_yolu(vaka_adi):
    for v in vakalari_listele():
        if os.path.basename(v["yol"]) == vaka_adi:
            return v["yol"]
    return None


def gercek_olcum(vaka_yolu):
    """Orijinal model + uzman etiketleriyle olcum (referans)."""
    v, f = obj_oku(vaka_yolu)
    _, _, etiket, _ = mesh_oku(vaka_yolu, os.path.splitext(vaka_yolu)[0] + ".json")
    return ko.ozet(ko.olc(v, FDI_DIZI[etiket], f))


def klinik_olcumler(kayit_dizin, g, vaka_adi=None):
    """Taramadan elde edilen model + yapay zeka etiketleriyle olcum ve gercek degerler (kayitta saklanir)."""
    paket = olcum_paketi(g["V"], g["F"], g["etiket"])
    yol = os.path.join(kayit_dizin, "olcum.json")
    gercek = None
    if os.path.exists(yol):
        try:
            kayit = json.load(open(yol, encoding="utf-8"))
            gercek = kayit.get("gercek") if kayit.get("surum") == ko.SURUM else None
        except Exception:
            gercek = None
    if gercek is None and vaka_adi:
        vy = _vaka_yolu(vaka_adi)
        if vy:
            gercek = gercek_olcum(vy)
            with open(yol, "w", encoding="utf-8") as fh:
                json.dump(dict(surum=ko.SURUM, gercek=gercek, tahmin=paket["ozet"]), fh, ensure_ascii=False, indent=1)
    if gercek:
        gercek = dict(gercek, genislik={int(k): v for k, v in gercek["genislik"].items()})
    return dict(tahmin=paket["ozet"], gercek=gercek, gorsel=paket["gorsel"], ham=paket["ham"])
