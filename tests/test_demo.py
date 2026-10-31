import numpy as np
import pytest

dash = pytest.importorskip("dash")


def test_arayuz_bilesenleri():
    import demo.app as a
    od = a.odontogram([11, 21, 36], {11: 0.91})
    assert od is not None
    assert a.asama_seridi("tarama") is not None
    V = np.random.rand(30, 3)
    F = np.array([[0, 1, 2], [2, 3, 4]])
    fig = a.seg_sekli({"V": V, "F": F, "etiket": np.zeros(30, dtype=int)})
    assert len(fig.data) == 1
    fig = a.yuzey_sekli({"V": V, "F": F, "sapma": np.zeros(30)})
    assert fig.data[0].cmax == pytest.approx(0.1)


def test_ust_cene_gorunumu():
    import demo.app as a
    V = np.zeros((200, 3)); V[:100, 2] = -5.0
    e = np.r_[np.ones(100), np.zeros(100)].astype(int)
    assert a._ust_cene_mi(V, e)
    V[:100, 2] = 5.0
    assert not a._ust_cene_mi(V, e)


def test_motor_durum_ve_durdurma():
    from demo.motor import TaramaMotoru
    m = TaramaMotoru()
    assert m.oku("asama")["asama"] is None
    assert not m.calisiyor
    m.durdur()


def test_kapanis_konumu_ust_ceneyi_alt_cenenin_ustune_oturtur():
    from demo.motor import kapanis_konumu
    rs = np.random.RandomState(0)
    # disler +z'ye bakan iki "cene": dis tepeleri z=10 civari, dis eti z=0 civari
    def cene(kayma):
        V = np.c_[rs.uniform(-20, 20, 4000), rs.uniform(-15, 15, 4000), rs.uniform(0, 10, 4000)] + kayma
        e = (V[:, 2] - kayma[2] > 5).astype(int) * 3
        return dict(V=V, etiket=e)
    alt, ust = cene(np.array([0.0, 0, 0])), cene(np.array([4.0, -2, 3]))
    R, t = kapanis_konumu(alt, ust, ortusme_mm=0.8)
    assert np.isclose(np.linalg.det(R), 1.0)               # ayna degil, gercek donme
    Vu = ust["V"] @ R.T + t
    du, da = Vu[ust["etiket"] > 0], alt["V"][alt["etiket"] > 0]
    assert du[:, 2].mean() > da[:, 2].mean()                 # ust cenenin disleri alt cenenin ustunde
    assert Vu[ust["etiket"] == 0, 2].mean() > du[:, 2].mean()  # ust cenenin dis eti en ustte (disler asagi bakar)
    assert np.allclose(du[:, :2].mean(0), da[:, :2].mean(0), atol=1e-6)
    assert np.isclose(np.percentile(da[:, 2], 99.5) - np.percentile(du[:, 2], 0.5), 0.8, atol=1e-6)


def test_hasta_ciftleri_ve_kapanis_taramalari_cokmez():
    from demo.motor import hasta_ciftleri, kapanis_taramalari
    c = hasta_ciftleri()
    assert isinstance(c, dict)
    for h, v in list(c.items())[:3]:
        assert "lower" in v and "upper" in v
        assert isinstance(kapanis_taramalari(h), list)


def test_kare_dosyasi_ilk_jpege_eklenen_veriye_dayanikli(tmp_path):
    """Bazi dosya aktarim araclari JPEG ile baslayan dosyaya meta veri (C2PA) ekliyor; kareler kaymamali."""
    import cv2
    from demo.motor import kareleri_kaydet, kareleri_oku, KARE_BASLIK
    kareler = [(i, cv2.imencode(".jpg", np.full((20, 30, 3), i * 40, np.uint8))[1].tobytes()) for i in range(4)]
    kareleri_kaydet(str(tmp_path), kareler)
    assert open(tmp_path / "kareler.bin", "rb").read(8) == KARE_BASLIK
    assert [b for _, b in kareleri_oku(str(tmp_path))] == [b for _, b in kareler]
    # eski bicim + ilk JPEG'in basligina 500 baytlik APP11 segmenti eklenmis hali
    ilk = kareler[0][1]
    ek = b"\xff\xeb" + (500).to_bytes(2, "big") + b"\x00" * 498
    bozuk = ilk[:20] + ek + ilk[20:]
    (tmp_path / "kareler.bin").write_bytes(bozuk + b"".join(b for _, b in kareler[1:]))
    okunan = kareleri_oku(str(tmp_path))
    assert [b for _, b in okunan[1:]] == [b for _, b in kareler[1:]]
    assert all(cv2.imdecode(np.frombuffer(b, np.uint8), 1) is not None for _, b in okunan)


def test_olcum_paneli_ve_cizgiler():
    import os
    os.environ.setdefault("DEMO_TARAYICI_ACMA", "1")
    from demo import app as A
    t = dict(cene="alt", genislik={41: 5.5, 31: 5.6, 46: 11.0}, ark_mevcut=70.0, ark_gerekli=72.5, ark_farki=2.5,
             interkanin=26.0, intermolar=45.0, uyari=None)
    g = dict(t, genislik={41: 5.4, 31: 6.4, 36: 11.2})
    panel = A.olcum_paneli(t, g)
    metin = str(panel)
    assert "hafif çapraşıklık" in metin and "fazla" in metin and "yok" in metin
    izler = A.olcum_cizgileri(dict(egri=np.zeros((5, 3)), cizgiler={41: (np.zeros(3), np.ones(3))}))
    assert len(izler) == 3


def test_motor_capraskilik_yorumu():
    from demo.motor import capraskilik_yorumu
    assert capraskilik_yorumu(None) == "hesaplanamadı"
    assert "boşluk" in capraskilik_yorumu(-3)
    assert capraskilik_yorumu(6) == "orta çapraşıklık"
    assert capraskilik_yorumu(10) == "şiddetli çapraşıklık"
