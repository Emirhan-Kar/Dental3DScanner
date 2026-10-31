import numpy as np
import pytest

from processing.klinik_olcum import olc, bolton, ozet, ark_sirasi, dis_maskeleri

GENISLIK_UST = {1: 8.6, 2: 6.6, 3: 7.6, 4: 7.0, 5: 6.6, 6: 10.2, 7: 9.2}
GENISLIK_ALT = {1: 5.4, 2: 5.9, 3: 6.9, 4: 7.0, 5: 7.1, 6: 11.2, 7: 10.5}


def _sentetik_cene(cene="ust", genislik=GENISLIK_UST, k=0.035, bosluk=0.0, tohum=0, kaydir=None):
    """Parabolik ark uzerine kutu seklinde disler. Disler arka arkaya (temasli) dizilir;
    'bosluk' her temasa eklenen aralik (mm)."""
    rs = np.random.RandomState(tohum)
    sira = [f for f in ark_sirasi(cene) if f % 10 <= 7]
    ws = [genislik[f % 10] for f in sira]
    toplam = sum(ws) + bosluk * (len(ws) - 1)
    # ark parametresi: y = -k x^2 ; yay uzunlugundan x'e sayisal donusum
    xs = np.linspace(-60, 60, 20001)
    ys = -k * xs ** 2
    s = np.r_[0, np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))]
    s -= s[len(s) // 2]
    V, E = [], []
    konum = -toplam / 2
    for f, w in zip(sira, ws):
        orta = konum + w / 2
        konum += w + bosluk
        i = np.searchsorted(s, orta)
        c = np.array([xs[i], ys[i]])
        t = np.array([1.0, -2 * k * xs[i]]); t /= np.linalg.norm(t)
        dik = np.array([-t[1], t[0]])
        if kaydir and f in kaydir:
            c = c + dik * kaydir[f]
        n = 2500
        u = rs.uniform(-w / 2, w / 2, n); v = rs.uniform(-3.5, 3.5, n); h = rs.uniform(0, 7, n)
        P = np.c_[c + np.outer(u, t) + np.outer(v, dik), h]
        V.append(P); E.append(np.full(n, f))
        g = np.c_[c + np.outer(rs.uniform(-w / 2, w / 2, 600), t) + np.outer(rs.uniform(-6, 6, 600), dik), rs.uniform(-4, -1, 600)]
        V.append(g); E.append(np.zeros(600, int))
    V = np.concatenate(V)
    # x: hastanin sagindan soluna. Ust cenede hastanin sagi (1. kadran) x<0 tarafinda olsun
    return V, np.concatenate(E)


def test_genislikleri_dogru_olcer():
    V, e = _sentetik_cene()
    o = olc(V, e)
    assert o["cene"] == "ust"
    for f, d in o["disler"].items():
        assert d["genislik"] == pytest.approx(GENISLIK_UST[f % 10], abs=0.6), f


def test_temasli_arkta_capraskilik_yok_bosluklu_arkta_negatif():
    o = olc(*_sentetik_cene())
    assert abs(o["ark"]["fark"]) < 1.5
    ob = olc(*_sentetik_cene(bosluk=1.0))
    assert ob["ark"]["fark"] < o["ark"]["fark"] - 5          # 10 temasta ~1 mm bosluk -> belirgin negatif


def test_ice_kacmis_dis_capraskilik_uretir():
    o = olc(*_sentetik_cene())
    # yan kesiciler ve kopekler arktan disari/ice itilmis: ark egrisi kisalir, gerekli alan ayni
    oc = olc(*_sentetik_cene(kaydir={12: -3.0, 22: -3.0, 13: 2.5, 23: 2.5}))
    assert oc["ark"]["gerekli"] == pytest.approx(o["ark"]["gerekli"], abs=1.5)
    assert oc["ark"]["fark"] > o["ark"]["fark"] + 0.8


def test_eksik_diste_gerekli_alan_yok_ve_uyari():
    V, e = _sentetik_cene()
    m = e != 14
    o = olc(V[m], e[m])
    assert o["ark"]["gerekli"] is None and 14 in o["ark"]["eksik"]
    assert o["uyari"]


def test_alt_cene_ve_bolton():
    ust = olc(*_sentetik_cene("ust", GENISLIK_UST))
    alt = olc(*_sentetik_cene("alt", GENISLIK_ALT, k=0.04, tohum=1))
    assert alt["cene"] == "alt"
    b = bolton(alt, ust)
    on_beklenen = 100 * sum(GENISLIK_ALT[k] for k in (1, 2, 3)) / sum(GENISLIK_UST[k] for k in (1, 2, 3))
    assert b["on"] == pytest.approx(on_beklenen, abs=2.0)
    assert ozet(alt)["genislik"]


def test_ucgenli_meshte_en_buyuk_parca_alinir():
    # iki ayri ucgen grubu ayni etikette: kucuk olan (yanlis etiket) atilmali
    V = np.array([[0, 0, 0], [5, 0, 0], [0, 5, 0], [5, 5, 0], [40, 40, 0], [40.5, 40, 0], [40, 40.5, 0]], float)
    F = np.array([[0, 1, 2], [1, 3, 2], [4, 5, 6]])
    e = np.full(len(V), 11)
    m = dis_maskeleri(V, e, F, min_alan_mm2=1.0, min_nokta=3)
    assert m[11][:4].all() and not m[11][4:].any()
