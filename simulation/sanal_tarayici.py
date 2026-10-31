"""
Sanal Agiz Ici Lazer Tarayici (Virtual Laser-Line Intraoral Scanner)

Gercek bir dis modelinin (Teeth3DS mesh) uzerinde, bir kameranin ve cizgi lazerin gorecegi
goruntuleri isin izleme (ray casting) ile uretir. Uretilen goruntuler projedeki GERCEK tarama
hattindan (LaserDetector -> Triangulator -> birlestirme -> mesh) gecirilir ve sonuc, kullanilan
gercek modelle (ground truth) karsilastirilir.

Optik geometri (kamera koordinatlari, OpenCV: x sag, y asagi, z ileri):
  * Pinhole kamera, K = [[fx,0,cx],[0,fy,cy],[0,0,1]]
  * Lazer yayici kameradan y ekseni boyunca 'baz_mm' kadar uzakta: E = (0, b, 0)
  * Lazer tabakasi kamera x eksenini icerir ve optik eksene 'aci' ile egilir:
        y*cos(a) + z*sin(a) - b*cos(a) = 0      -> goruntude yatay bir cizgi
    Optik ekseni z = b / tan(a) mesafesinde keser (calisma mesafesi).
"""
import numpy as np
import open3d as o3d
import cv2


class SanalKamera:
    def __init__(self, genislik=640, yukseklik=480, fx=600.0, fy=600.0, cx=None, cy=None,
        baz_mm=8.0, calisma_mesafesi_mm=17.0):
        self.w, self.h = int(genislik), int(yukseklik)
        self.fx, self.fy = float(fx), float(fy)
        self.cx = (self.w - 1) / 2.0 if cx is None else float(cx)
        self.cy = (self.h - 1) / 2.0 if cy is None else float(cy)
        self.b = float(baz_mm)
        self.alfa = float(np.arctan2(self.b, calisma_mesafesi_mm))
        self.calisma_mesafesi = float(calisma_mesafesi_mm)

    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64)

    @property
    def lazer_duzlemi(self):
        """Kamera koordinatinda [A, B, C, D] (Triangulator ile ayni format)."""
        a = self.alfa
        return np.array([0.0, np.cos(a), np.sin(a), -self.b * np.cos(a)])

    @property
    def yayici(self):
        return np.array([0.0, self.b, 0.0])

    def isin_yonleri(self):
        """Her piksel icin kamera koordinatinda (normalize olmayan) isin yonu (H*W, 3)."""
        u, v = np.meshgrid(np.arange(self.w), np.arange(self.h))
        d = np.stack([(u - self.cx) / self.fx, (v - self.cy) / self.fy, np.ones_like(u, dtype=float)], -1)
        return d.reshape(-1, 3)


class SanalTarayici:
    """Bir mesh uzerinde lazer cizgisi goruntuleri ureten isin izleyici."""

    def __init__(self, koseler, yuzler, kose_etiketleri=None, kamera=None,
        cizgi_kalinligi_mm=0.05, ortam_isigi=0.22, goruntu_gurultusu=3.0, tohum=0):
        self.kamera = kamera or SanalKamera()
        self.koseler = np.asarray(koseler, dtype=np.float64)
        self.yuzler = np.asarray(yuzler, dtype=np.int64)
        mesh = o3d.t.geometry.TriangleMesh()
        mesh.vertex.positions = o3d.core.Tensor(self.koseler.astype(np.float32))
        mesh.triangle.indices = o3d.core.Tensor(self.yuzler.astype(np.int32))
        self.sahne = o3d.t.geometry.RaycastingScene()
        self.sahne.add_triangles(mesh)
        # Yuz renkleri: dis (mine) beyazimsi, dis eti pembe-kirmizi (kirmizi lazerle karisabilir!)
        if kose_etiketleri is not None:
            dis = np.asarray(kose_etiketleri)[self.yuzler].max(axis=1) > 0
        else:
            dis = np.ones(len(self.yuzler), dtype=bool)
        self.yuz_renk = np.where(dis[:, None], [[0.86, 0.88, 0.90]], [[0.55, 0.42, 0.80]])  # BGR
        self.cizgi_sigma = float(cizgi_kalinligi_mm)
        self.ortam = float(ortam_isigi)
        self.gurultu = float(goruntu_gurultusu)
        self.rng = np.random.RandomState(tohum)
        self._yonler = self.kamera.isin_yonleri()

    def kare_uret(self, T_dk):
        """Kamera pozu T_dk (kamera->dunya, 4x4) icin BGR goruntu (uint8) uretir.

        Returns:
            goruntu (H, W, 3) uint8, bilgi dict (derinlik haritasi, lazer maskesi)
        """
        K = self.kamera
        R, t = T_dk[:3, :3], T_dk[:3, 3]
        yon_d = self._yonler @ R.T
        isinlar = np.hstack([np.repeat(t[None], len(yon_d), 0), yon_d]).astype(np.float32)
        sonuc = self.sahne.cast_rays(o3d.core.Tensor(isinlar))
        t_hit = sonuc["t_hit"].numpy()
        vurdu = np.isfinite(t_hit)
        prim = sonuc["primitive_ids"].numpy()
        normal = sonuc["primitive_normals"].numpy()

        P = np.zeros((len(yon_d), 3))
        P[vurdu] = t + yon_d[vurdu] * t_hit[vurdu, None]

        # Ortam / flas aydinlatmasi (Lambert, kamera yonunden)
        goz = -yon_d / np.linalg.norm(yon_d, axis=1, keepdims=True)
        lam = np.zeros(len(yon_d))
        lam[vurdu] = np.abs(np.sum(normal[vurdu] * goz[vurdu], axis=1))
        renk = np.zeros((len(yon_d), 3))
        renk[vurdu] = self.yuz_renk[prim[vurdu]] * (0.35 + 0.65 * lam[vurdu, None]) * self.ortam

        # Lazer tabakasi (dunya koordinatinda)
        A, B, C, D = K.lazer_duzlemi
        n_k = np.array([A, B, C])
        n_d = R @ n_k
        d_d = D - n_d @ t                            # n.x + d = 0 (dunya)
        E = R @ K.yayici + t
        mesafe = np.zeros(len(yon_d))
        mesafe[vurdu] = (P[vurdu] @ n_d + d_d) / np.linalg.norm(n_d)
        aday = vurdu & (np.abs(mesafe) < 4 * self.cizgi_sigma)
        lazer = np.zeros(len(yon_d))
        if aday.any():
            idx = np.where(aday)[0]
            EP = P[idx] - E
            uz = np.linalg.norm(EP, axis=1)
            yon = EP / uz[:, None]
            golge = self.sahne.cast_rays(o3d.core.Tensor(np.hstack([np.repeat(E[None], len(idx), 0), yon]).astype(np.float32)))
            gorunur = golge["t_hit"].numpy() > uz - 0.02           # yayicidan noktaya engelsiz
            cos_l = np.abs(np.sum(normal[idx] * -yon, axis=1))
            lazer[idx] = gorunur * np.exp(-0.5 * (mesafe[idx] / self.cizgi_sigma) ** 2) * (0.3 + 0.7 * cos_l)

        renk[:, 2] += 1.0 * lazer        # R kanali (BGR)
        renk[:, 1] += 0.12 * lazer
        renk[:, 0] += 0.12 * lazer
        img = (np.clip(renk, 0, 1) * 255).reshape(K.h, K.w, 3)
        img = cv2.GaussianBlur(img, (0, 0), 0.6)
        if self.gurultu > 0:
            img = img + self.rng.normal(0, self.gurultu, img.shape)
        goruntu = np.clip(img, 0, 255).astype(np.uint8)
        derinlik = np.full(len(yon_d), np.nan)
        derinlik[vurdu] = t_hit[vurdu]           # yon_d'nin z bileseni 1 oldugu icin t = kamera Z'si
        return goruntu, {"derinlik": derinlik.reshape(K.h, K.w), "lazer": lazer.reshape(K.h, K.w)}


# ───────────────────────── tarama yolu ─────────────────────────

def _yumusat(P, pencere=7):
    if len(P) < pencere:
        return P
    k = np.ones(pencere) / pencere
    ped = np.pad(P, ((pencere // 2, pencere // 2), (0, 0)), mode="edge")
    return np.stack([np.convolve(ped[:, i], k, mode="valid") for i in range(P.shape[1])], 1)


def ark_egrisi(koseler, etiket_idx, fdi_listesi, adim_mm=0.25, yukari=(0, 0, 1)):
    """Dis merkezlerinden gecen, ark boyunca esit aralikli yumusak egri.

    Returns:
        noktalar (M,3) — dis merkezlerinin yatay konumu, 'yukari' yonunde dis tepesi yuksekligi; teget (M,3)
    """
    yukari = np.asarray(yukari, dtype=float)
    yukari = yukari / np.linalg.norm(yukari)
    merkezler = []
    for idx, fdi in enumerate(fdi_listesi):
        if idx == 0:
            continue
        m = etiket_idx == idx
        if m.sum() > 50:
            q, i = divmod(fdi, 10)
            c = koseler[m].mean(0)
            tepe = (koseler[m] @ yukari).max()
            merkezler.append((fdi, c + yukari * (tepe - c @ yukari)))
    if len(merkezler) < 3:
        raise ValueError("Ark egrisi icin yeterli dis yok.")
    cene_alt = merkezler[0][0] // 10 in (3, 4)
    sag = 4 if cene_alt else 1

    def sira(f):
        q, i = divmod(f, 10)
        return -i if q == sag else i
    merkezler.sort(key=lambda m: sira(m[0]))
    P = np.array([m[1] for m in merkezler])
    # uclari biraz uzat (son dislerin distal yuzu de taransin)
    P = np.vstack([P[0] + (P[0] - P[1]) * 0.5, P, P[-1] + (P[-1] - P[-2]) * 0.5])
    # yogun ornekle
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    ss = np.arange(0, s[-1], adim_mm)
    Q = np.stack([np.interp(ss, s, P[:, i]) for i in range(3)], 1)
    Q = _yumusat(Q, pencere=max(3, int(4.0 / adim_mm) | 1))
    T = np.gradient(Q, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    return Q, T


def tarama_pozlari(ark, teget, yukari, egim_derece=0.0, hedef_derinlik_mm=2.0, mesafe_mm=17.0):
    """Ark boyunca kamera pozlari (kamera->dunya).

    Kamera optik ekseni (z) asagi (dislere) bakar; kamera y ekseni hareket yonu (teget),
    boylece yatay lazer cizgisi ark boyunca suputurulur. 'egim_derece' ile kamera teget
    ekseni etrafinda yanak (+) / dil (-) tarafina egilir.
    """
    pozlar = []
    for p, tg in zip(ark, teget):
        tg = tg - np.dot(tg, yukari) * yukari
        tg /= np.linalg.norm(tg)
        z = -yukari
        x = np.cross(tg, z)
        R0 = np.stack([x, tg, z], 1)
        a = np.radians(egim_derece)
        Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        R = R0 @ Ry
        hedef = p - yukari * hedef_derinlik_mm
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = hedef - R[:, 2] * mesafe_mm
        pozlar.append(T)
    return pozlar
