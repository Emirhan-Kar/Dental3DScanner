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
- [Docker ile Çalıştırma](#5--docker-ile-kurulum-ve-çalıştırma)
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
│   ├── 2DTeethData/           # Kalibrasyon ve simülasyon görselleri
│   ├── TestVideo/             # Test video dosyaları
│   └── four/                  # Ek referans veriler
│
├── calibration/               # Optik ve Geometrik Kalibrasyon
│   ├── calib_script.py        # Chessboard köşe tespiti ve matris optimizasyonu
│   └── camera_params.yaml     # K matrisi ve distorsiyon katsayıları
│
├── processing/                # Temel Görüntü İşleme ve Rekonstrüksiyon
│   ├── laser_detector.py      # HSV maskeleme ve alt-piksel CoM merkez çıkarıcı
│   ├── triangulation.py       # 2D → 3D lazer düzlem kesişim koordinat dönüştürücü
│   ├── icp_stitcher.py        # RANSAC Global + Point-to-Plane ICP hizalama motoru
│   ├── mesh_generator.py      # Poisson & Ball-Pivoting yüzey rekonstrüksiyon modülü
│   └── filters.py             # SOR, ROR, Bounding Box ve bütünleşik filtreleme boru hattı
│
├── ai_model/                  # 3D Derin Öğrenme Modülü
│   ├── dataset.py             # PyTorch 3D veri yükleyici (Teeth3DS) ve augmentasyon
│   ├── network.py             # DentalPointNetSeg segmentasyon ağ mimarisi
│   ├── train.py               # Model eğitim döngüsü (Adam + StepLR + Checkpoint)
│   ├── inference.py           # Nihai mesh üzerinde FDI segmentasyon ve etiketleme
│   ├── metrics.py             # IoU, mIoU, Dice segmentasyon metrikleri
│   └── best_dental_model.pth  # Eğitilmiş model ağırlıkları (~5.3 MB)
│
├── ui/                        # Görselleştirme Katmanı
│   └── viewer.py              # Open3D 3D sahne + FDI renk paletli segmentasyon görselleştirici
│
├── utils/                     # Yardımcı Altyapı Modülleri
│   ├── config.py              # YAML konfigürasyon yükleyici (ConfigNode dot-notation)
│   ├── logger.py              # Rotasyonlu dosya + Unicode-safe konsol logger
│   ├── validator.py           # Mesh, JSON, PLY, görsel doğrulayıcı + veri seti tarayıcı
│   └── video_stream.py        # Çoklu iş parçacıklı asenkron video yakalama
│
├── configs/                   # Merkezi Konfigürasyon
│   └── config.yaml            # Tüm modül parametreleri (kamera, lazer, ICP, AI, mesh, log)
│
├── tests/                     # Birim ve Entegrasyon Testleri (86 test)
│   ├── test_ai_model.py       # PointNet forward pass ve boyut doğrulaması
│   ├── test_calibration.py    # Kalibrasyon parametre doğrulama
│   ├── test_filters.py        # SOR, ROR, crop filtreleme testleri
│   ├── test_icp_stitcher.py   # ICP hizalama, FPFH öznitelik testleri
│   ├── test_laser_detector.py # Alt-piksel tespit, HSV maskeleme testleri
│   ├── test_mesh_generator.py # Poisson, BPA rekonstrüksiyon testleri
│   ├── test_metrics.py        # IoU, mIoU, Dice metrik testleri
│   ├── test_pipeline.py       # End-to-end pipeline entegrasyon testi
│   ├── test_triangulation.py  # 2D→3D dönüşüm, YAML yükleme testleri
│   ├── test_utils.py          # Config, logger yardımcı testleri
│   ├── test_validator.py      # Mesh, annotation doğrulama testleri
│   └── test_video_stream.py   # Asenkron video akışı testleri
│
├── output/                    # Üretilen 3D Model Çıktıları
│   ├── final_model.stl        # Son katı diş modeli
│   ├── final_scan.ply         # Son tarama nokta bulutu
│   └── ...                    # Test çıktıları (.ply, .stl)
│
├── logs/                      # Uygulama Log Dosyaları
│   └── dental_scanner.log     # Rotasyonlu merkezi log
│
├── main.py                    # Pipeline ana çalıştırma betiği (orkestratör)
├── requirements.txt           # Bağımlılıklar
├── Dockerfile                 # Konteyner imaj tanımı (OpenCV, Open3D, PyTorch)
├── docker-compose.yml         # Çoklu servis orkestrasyonu (app, test, ai-train, ai-infer)
├── .dockerignore              # Docker derleme bağlamı filtre kuralları
├── pytest.ini                 # Test konfigürasyonu
├── LICENSE                    # Akademik kullanım lisansı
├── DATA_LICENSE.md            # Teeth3DS veri lisansı (CC BY-NC-ND 4.0) ve atıflar
└── README.md                  # Teknik dokümantasyon
```

---

## 🚀 Kurulum ve Çalıştırma

### 1. Depoyu Klonlayın
```bash
git clone https://github.com/Emirhan-Kar/Dental3DScanner.git
cd Dental3DScanner
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

### 4. GPU Desteği (Opsiyonel)
PyTorch CUDA desteği için [PyTorch resmi sitesinden](https://pytorch.org/get-started/locally/) sisteminize uygun komutu çalıştırın:
```bash
# Örnek: CUDA 12.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 5. 🐳 Docker ile Kurulum ve Çalıştırma

Projeyi sisteminize hiçbir kütüphane kurmadan Docker ve Docker Compose ile izole bir ortamda çalıştırabilirsiniz.

#### A. Docker İmajını Oluşturma
```bash
docker build -t dental-scanner:latest .
```

#### B. Docker Compose ile Hızlı Başlatma
```bash
# 1. Simülasyonu çalıştırıp çıktı üretme (.stl ve .ply)
docker compose up app

# 2. Tüm birim ve entegrasyon testlerini çalıştırma (86 test)
docker compose run --rm test

# 3. AI model eğitimini başlatma
docker compose up ai-train

# 4. 3D diş segmentasyonu çıkarımı yapma
docker compose run --rm ai-infer
```

#### C. Doğrudan `docker run` ile Çalıştırma
```bash
# Simülasyon çalıştırma ve çıktıları yerel output klasörüne alma:
docker run --rm -v "%cd%\output:/app/output" dental-scanner:latest

# Konteyner içinde testleri koşturma:
docker run --rm dental-scanner:latest pytest tests/ -v
```

---

## 🖥️ Demo Arayüzü (Komisyon Sunumu)

İnternet gerektirmeyen, tarayıcıda açılan yerel bir arayüz:

```bash
# Windows: proje klasöründe demo_baslat.bat dosyasına çift tıklayın, ya da:
venv_gpu\Scripts\python.exe demo\app.py
# Tarayıcıda otomatik açılır: http://127.0.0.1:8050
```

| Sekme | Ne gösterir |
|---|---|
| **Canlı sanal tarama** | Hasta modeli ve tarama ayarları seçilir; sanal kamera görüntüsü (lazer + tespit), büyüyen 3D nokta bulutu, hat aşamaları, ardından sapma haritası, renkli diş segmentasyonu, FDI diş şeması, doğruluk ölçümleri ve otomatik klinik ölçümler (diş genişlikleri, ark uzunluğu, çapraşıklık; uzman etiketleriyle yan yana) |
| **İki çene birlikte** | Aynı hastanın alt ve üst çenesi tek sahnede (yaklaşık kapanış, açılıp kapanabilir); iki çenenin ölçümleri ve Bolton analizi |
| **Dosyadan segmentasyon** | Herhangi bir .obj / .ply / .stl çene modeli yüklenir, DGCNN her dişi FDI numarasıyla işaretler ve klinik ölçümleri çıkarır |
| **Deney sonuçları** | `rapor_gorselleri/` altındaki tüm şekiller |

- Bir ayar kombinasyonu bir kez tarandıktan sonra `runs/demo/` altına kaydedilir. Aynı ayarlar tekrar seçildiğinde birkaç saniyede kayıttan oynatılır (sunumda bekleme olmaz). Canlı taramayı zorlamak için "Kayıtlı tarama varsa hızlı oynat" kutusunu kaldırın.
- Önerilen sunum ayarı: 3 geçiş, 0,5 mm aralık (~2–3 dk canlı; yüzey sapması ~0,036 mm, segmentasyon mIoU ~0,93).
- GPU varsa segmentasyon otomatik olarak CUDA ile çalışır.

### Otomatik klinik ölçümler (`processing/klinik_olcum.py`)

Segmentasyondan ortodontik ölçümler: her dişin meziodistal genişliği, mevcut ark uzunluğu (iki taraftaki 1. büyük azı dişlerinin meziyalleri arası, dişlerden geçen düzgün ark eğrisi boyunca), gerekli alan (5–5 dişlerinin genişlik toplamı), ark uzunluğu farkı (çapraşıklık / boşluk), interkanin ve intermolar genişlik, iki çene için Bolton oranları. Doğrulama:

```bash
python experiments/klinik_olcum_deneyi.py --liste secim.json          # test seti: yapay zekâ vs. uzman etiketi
python experiments/klinik_olcum_deneyi.py --sadece-ciz                # şekil 18'i yeniden çiz
```

---

## 🎮 Kullanım Örnekleri

### Sentetik Simülasyon (Donanım Gerektirmez)
Fiziksel kamera veya lazer olmadan pipeline'ı doğrulamak için sentetik diş arkı simülasyonu:
```bash
python main.py --simulation --show-2d --show-3d
```

### Video Dosyasından Tarama
```bash
python main.py --input data/TestVideo/scan.mp4 --output output/scan_result.ply --show-2d
```

### Canlı Kamera ile Tarama
```bash
python main.py --input 0 --show-2d --show-3d --export-mesh output/dental_model.stl
```

### STL/OBJ Katı Model Üretimi
```bash
python main.py --simulation --export-mesh output/dental_model.stl --mesh-method poisson
```

### Tüm CLI Parametreleri
```bash
python main.py --help
```

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `--input` | `None` | Video dosyası yolu veya kamera indeksi (ör: `0`) |
| `--output` | `output/scan_result.ply` | Çıktı nokta bulutu dosyası (.ply) |
| `--export-mesh` | `None` | Katı model çıktısı (.stl veya .obj) |
| `--mesh-method` | `poisson` | Yüzey rekonstrüksiyon yöntemi (`poisson` / `ball_pivoting`) |
| `--show-2d` | `False` | 2D lazer algılama monitor penceresi |
| `--show-3d` | `False` | Tarama sonrası 3D Open3D görselleştirme |
| `--simulation` | `False` | Sentetik diş tarama simülasyonu |
| `--max-frames` | `None` | Maksimum işlenecek kare sayısı |
| `--no-threaded` | `False` | Asenkron video okuyucuyu devre dışı bırak |
| `--config` | `configs/config.yaml` | Özel konfigürasyon dosyası yolu |

---

## 🧠 AI Model Eğitimi ve Çıkarımı

### Model Eğitimi
```bash
python ai_model/train.py --epochs 120 --batch-size 4 --lr 0.001 --num-points 4096
```

### Eğitime Devam Etme (Checkpoint Resume)
```bash
python ai_model/train.py --resume ai_model/best_dental_model.pth --epochs 200
```

### 3D Diş Segmentasyonu Çıkarımı
```bash
python ai_model/inference.py --mesh data/3DTeethData/data_part_1/ornek/model.obj --output-ply output/segmented.ply
```

### Headless Mod (GUI Olmadan)
```bash
python ai_model/inference.py --mesh model.obj --no-gui --output-ply output/result.ply
```

---

## ⚙️ Kamera Kalibrasyonu

Chessboard kalibrasyon görselleri ile kamera iç parametrelerini çıkarmak için:
```bash
python calibration/calib_script.py --folder data/2DTeethData --cols 9 --rows 6 --square-size 25.0
```

Çıktı: `calibration/camera_params.yaml` dosyasına K matrisi ve distorsiyon katsayıları kaydedilir.

---

## 🧪 Testleri Çalıştırma

### Tüm Testleri Çalıştır
```bash
python -m pytest tests/ -v
```

### Belirli Bir Modülü Test Et
```bash
python -m pytest tests/test_ai_model.py -v
python -m pytest tests/test_pipeline.py -v
```

### Kısa Özet Rapor
```bash
python -m pytest tests/ --tb=short -q
```

> **Not:** Test suite'inde toplam **86 birim ve entegrasyon testi** bulunmaktadır.

---

## 📦 Veri Seti ve Test Verilerinin Temini

Büyük boyutlu 3D diş tarama modelleri ve test verileri (`~16.25 GB`) dosya boyutu sınırları nedeniyle Git deposuna dahil edilmemiştir ve bulut depolama üzerinden paylaşılmaktadır.

1. **Veri Setini İndirin:**
   * 🔗 **Google Drive İndirme Bağlantısı:** [Dental_3DTeeth_Dataset.zip İndir (Google Drive)](https://drive.google.com/file/d/1lU2RUMmP984PJrWWGSY4w8_HX1v1zkAD/view?usp=drive_link)
   * **Arşiv Adı:** `Dental_3DTeeth_Dataset.zip` (~16.25 GB)

2. **Arşivi Çıkarın:**
   İndirdiğiniz `Dental_3DTeeth_Dataset.zip` arşivini proje kök dizinindeki `data/` klasörünün içerisine çıkartın. Çıkarma işleminden sonra dizin yapısı şu şekilde olmalıdır:
   ```text
   data/
   ├── 3DTeethData/
   │   ├── data_part_1/
   │   └── data_part_2/
   ├── 2DTeethData/
   ├── TestVideo/
   └── four/
   ```

3. **Veri Seti Bütünlüğünü Doğrulayın:**
   ```bash 
   python utils/validator.py --data-root data/3DTeethData --parts data_part_1 data_part_2
   ```

---

## 📜 Lisans

**Copyright (c) 2026 Emirhan Kar & Furkan Genç**. Tüm hakları saklıdır.

- **Kod, model ağırlıkları, rapor ve şekiller:** [DentalScanner Akademik Kullanım Lisansı](LICENSE). Yalnızca ticari olmayan akademik amaçlarla (araştırma, eğitim, tez, yayın) ve atıf yapılarak kullanılabilir. Ticari kullanım, klinik kullanım ve diğer tüm kullanımlar yasaktır; bunlar için yazarların yazılı izni gerekir.
- **Veri seti:** Teeth3DS üçüncü taraf verisidir, [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) lisanslıdır ve bu depoda yer almaz. Ayrıntılar ve gerekli atıflar için [DATA_LICENSE.md](DATA_LICENSE.md) dosyasına bakın.

> ⚠️ Bu yazılım bir araştırma prototipidir; gerçek hastalarda teşhis, tedavi ya da planlama amacıyla kullanılamaz.
