# 🦷 AI-Driven Intraoral 3D Reconstruction & Dental Segmentation Engine

Bu depo, video akışı üzerinden yüzey profili çıkarımı, gerçek zamanlı 3D nokta bulutu hizalama (ICP) ve derin öğrenme tabanlı diş segmentasyonu gerçekleştiren uçtan uca bir **3D Rekonstrüksiyon ve Analiz Yazılım Motoru**dur.

Sistem, fiziksel cihaz bağımlılığını en aza indirip ağır iş yükünü matematiksel optimizasyon ve bilgisayarlı görü algoritmalarıyla çözmeyi amaçlar.

---

## 📌 İçindekiler
- [Yazılım Mimarisi](#-yazılım-mimarisi-ve-veri-akış-şeması)
- [Matematiksel ve Algoritmik Temeller](#-matematiksel-ve-algoritmik-temeller)
- [3D Derin Öğrenme Modülü](#-3d-derin-öğrenme-modülü-tooth-segmentation)
- [Dizin Mimarisi](#-dizin-mimarisi)
- [Kurulum ve Çalıştırma](#-kurulum-ve-çalıştırma)
- [Veri Seti ve Test Verileri](#-veri-seti-ve-test-verilerinin-temini)

---

## 🏗️ Yazılım Mimarisi ve Veri Akış Şeması

```text
[ Ham Video / Kare Akışı ]
           │
           ▼
[ 1. Ön İşleme & Maskeleme (OpenCV) ] ──► Renk Filtresi & Alt-Piksel Tepe Tespiti
           │
           ▼
[ 2. Triangulation (Üçgenleme Motoru) ] ──► Projeksiyon Matematiği -> 3D Noktalar (X, Y, Z)
           │
           ▼
[ 3. Filtreleme & Downsampling ] ──► VoxelGrid & Statistical Outlier Removal
           │
           ▼
[ 4. Canlı ICP Hizalama (Open3D) ] ──► Point-to-Plane ICP + Pose Graph Optimization
           │
           ▼
[ 5. Yüzey Örme (Mesh Generation) ] ──► Poisson Surface Reconstruction (.ply / .stl)
           │
           ▼
[ 6. AI 3D Segmentasyon (PyTorch) ] ──► FDI Standartlarında 32 Diş Tespiti & Analizi
```

---

## 🔬 Matematiksel ve Algoritmik Temeller

### 1. Alt-Piksel Çizgi Tespiti (Sub-pixel Peak Detection)
Işık profilinin düştüğü piksel sütunlarında çizginin tepe noktası sadece tam sayı piksellerle sınırlı kalmaz. Enerji dağılımının ağırlık merkezi (Center of Mass) veya 1D Gaussian eğrisi uydurularak alt-piksel hassasiyeti elde edilir:

$$c = \frac{\sum_{i} i \cdot I(i)}{\sum_{i} I(i)}$$

### 2. Üçgenleme ve Koordinat Dönüşümü (Triangulation)
Kamera koordinat sistemindeki her bir $p = [u, v, 1]^T$ homojen pikseli ile kalibre edilmiş projeksiyon düzlemi kesiştirilerek metrik uzaydaki 3D koordinatlar ($X, Y, Z$) hesaplanır:

$$Z \cdot \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K \cdot \begin{bmatrix} X \\ Y \\ Z \end{bmatrix}$$

Burada $K$, odak uzaklıkları ($f_x, f_y$) ve optik merkezi ($c_x, c_y$) içeren iç parametre matrisidir.

### 3. Point-to-Plane ICP (Yinelemeli En Yakın Nokta)
Ardışık gelen nokta bulutu parçalarını birleştirmek için standart nokta-nokta mesafesi yerine, yüzey normallerini ($n_i$) kullanan **Point-to-Plane** hata fonksiyonu minimize edilir:

$$E(R, t) = \sum_{i} \left( \left( R p_i + t - q_i \right) \cdot n_i \right)^2$$

Bu yaklaşım, diş gibi pürüzsüz ve kavisli yüzeylerde kaymayı (drift) minimuma indirir.

---

## 🧠 3D Derin Öğrenme Modülü (Tooth Segmentation)

Yapay zeka katmanı 2D görüntüler yerine doğrudan **3D nokta bulutu ve poligon ağları (mesh)** üzerinde çalışır:

* **Girdi Formatı:** Rekonstrüksiyon sonucu üretilen global 3D mesh ($N \times 6$ matrisi: $X, Y, Z$ koordinatları + Yüzey Normalleri $N_x, N_y, N_z$).
* **Mimari:** PointNet++ / MeshSegNet hiyerarşik öznitelik çıkarıcılar.
* **Sınıflandırma Standardı:** FDI World Dental Federation Two-Digit System (11–48 arası 32 diş sınıfı + Gingiva/Diş Eti sınıfı).
* **Boşluk & Morfoloji Yönetimi:** Eksik diş bölgelerinde diş eti dokusunun geometrisini korur; semantik etiketleme sayesinde boşluğun komşuluk ilişkilerini doğru indeksler.

---

## 📂 Dizin Mimarisi

```text
intraoral-3d-engine/
│
├── data/                      # Test verileri ve 3D modeller
│   ├── 3DTeethData/           # Açık kaynaklı referans diş modelleri (.obj, .ply)
│   ├── 2DTeethData/           # Simülasyon test videoları
│   └── TestVideo/             # Test Videoları için
│
├── calibration/               # Optik ve Geometrik Kalibrasyon
│   ├── calib_script.py        # Chessboard köşe tespiti ve matris optimizasyonu
│   └── camera_params.yaml     # K matrisi ve distorsiyon katsayıları
│
├── processing/                # Temel Görüntü İşleme ve Rekonstrüksiyon
│   ├── laser_detector.py      # Maskeleme ve alt-piksel merkez çıkarıcı
│   ├── triangulation.py       # 2D -> 3D koordinat dönüştürücü
│   └── icp_stitcher.py        # Open3D tabanlı Point-to-Plane ICP motoru
│
├── ai_model/                  # 3D Derin Öğrenme Modülü
│   ├── dataset.py             # PyTorch 3D veri yükleyici ve augmentasyon
│   ├── network.py             # 3D segmentasyon ağ mimarisi
│   ├── train.py               # Model eğitim döngüsü
│   └── inference.py           # Nihai mesh üzerinde segmentasyon ve etiketleme
│
├── ui/                        # Görselleştirme Katmanı
│   └── viewer.py              # Open3D gerçek zamanlı sahne görselleştirici
│
├── main.py                    # Pipeline ana çalıştırma betiği
├── requirements.txt           # Bağımlılıklar
└── README.md                  # Teknik dokümantasyon
```

---

## 🚀 Kurulum ve Çalıştırma

### 1. Depoyu Klonlayın
```bash
git clone [https://github.com/KULLANICI_ADIN/intraoral-3d-engine.git](https://github.com/KULLANICI_ADIN/intraoral-3d-engine.git)
cd intraoral-3d-engine
```

### 2. Sanal Ortam Kurulumu
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Bağımlılıkların Yüklenmesi
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📊 Veri Seti ve Test Verilerinin Temini

Büyük boyutlu 3D tarama dosyaları (`.obj`, `.ply`) Git deposuna dahil edilmemiştir.

1. Test modellerini içeren paketi bulut depolama üzerinden temin edin:
   * **İndirme Bağlantısı:** `[BULUT_DEPOLAMA_LINKI]`
2. İndirilen dosyaları `data/raw_meshes/` dizinine yerleştirin.
