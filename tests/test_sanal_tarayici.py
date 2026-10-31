import numpy as np
import pytest

from simulation.sanal_tarayici import SanalTarayici, SanalKamera, tarama_pozlari
from processing.laser_detector import LaserDetector
from processing.triangulation import Triangulator


def _duzlem(boyut=40.0, n=40):
    x, y = np.meshgrid(np.linspace(-boyut / 2, boyut / 2, n), np.linspace(-boyut / 2, boyut / 2, n))
    v = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    f = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            f += [[a, a + 1, a + n], [a + 1, a + n + 1, a + n]]
    return v, np.array(f)


@pytest.mark.parametrize("mesafe", [14.0, 17.0, 20.0])
def test_duz_yuzey_derinligi(mesafe):
    """Dik bakan kamerayla duz yuzey: gercek hat (dedektor + ucgenleme) yuzeyi z=0'da bulmali."""
    v, f = _duzlem()
    kam = SanalKamera()          # projede kullanilan kamera: 640x480, f=600 px
    st = SanalTarayici(v, f, kamera=kam, goruntu_gurultusu=2.0)
    T = tarama_pozlari(np.array([[0.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]]), np.array([0, 0, 1.0]),
                       hedef_derinlik_mm=0.0, mesafe_mm=mesafe)[0]
    img, bilgi = st.kare_uret(T)
    tepeler, _ = LaserDetector().detect_subpixel_peaks(img)
    assert len(tepeler) > 200                       # cizgi goruntunun buyuk kisminda gorunmeli
    P = Triangulator(camera_matrix=kam.K, laser_plane=kam.lazer_duzlemi).pixels_to_3d(tepeler)
    Pd = P @ T[:3, :3].T + T[:3, 3]
    # Agirlik merkezi yonteminin bilinen "piksel kilitlenmesi" (pixel locking) etkisi: cizginin piksel
    # izgarasina gore konumuna bagli ~0.02 mm'ye kadar sistematik sapma olabilir.
    assert np.median(np.abs(Pd[:, 2])) < 0.03       # 30 mikron
    assert np.percentile(np.abs(Pd[:, 2]), 95) < 0.05


def test_lazer_duzlemi_calisma_mesafesi():
    kam = SanalKamera(baz_mm=8.0, calisma_mesafesi_mm=17.0)
    A, B, C, D = kam.lazer_duzlemi
    # optik eksen (x=y=0) ile kesisim z = -D / C
    assert -D / C == pytest.approx(17.0, rel=1e-6)
    # yayici duzlem uzerinde
    assert B * kam.b + D == pytest.approx(0.0, abs=1e-9)
