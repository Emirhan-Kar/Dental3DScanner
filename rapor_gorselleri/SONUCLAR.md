# Yapay Zeka Segmentasyon Sonuçları (Rapor İçin Özet)

Eğitim: Google Colab, NVIDIA A100, 29.10.2026. Ham çıktılar: `runs/` klasörü.

## Veri seti ve bölme

- Teeth3DS: **900 hasta, 2.100 çene taraması** (alt + üst çene, 7 parça).
- **Hasta bazlı** bölme (aynı hastanın alt ve üst çenesi her zaman aynı sette):

| Set | Hasta | Tarama |
|---|---|---|
| Eğitim | 720 | 1.680 |
| Doğrulama | 90 | 213 |
| Test | 90 | 207 |

- Bölme `runs/dgcnn/split.json` dosyasında; iki model de aynı test hastalarında ölçüldü.
- Sınıflar: 32 diş (FDI 11–48) + diş eti = 33 sınıf.

## Ana sonuç tablosu (test seti, tüm köşeler üzerinde)

| Model | Nokta doğruluğu | mIoU (dişler) | Dice | Diş tanıma (IoU > 0,5) | Tarama başına mIoU (ort. ± ss) |
|---|---|---|---|---|---|
| Eski PointNet (önceki sürüm) | 0,459 | 0,000 | 0,019 | %0,0 | 0,000 |
| PointNet (yeni eğitim) | 0,894 | 0,732 | 0,842 | %93,0 | 0,797 ± 0,120 |
| **DGCNN (yeni eğitim)** | **0,931** | **0,817** | **0,896** | **%96,5** | **0,868 ± 0,114** |

- DGCNN, PointNet'e göre mIoU'da **+8,5 puan** (0,732 → 0,817) daha iyi. Fark, DGCNN'in her noktanın k=20 en yakın komşusundan yerel geometri öğrenmesinden geliyor.
- DGCNN'de tarama başına mIoU medyanı **0,901**. 207 taramanın sadece 5'inde 0,5'in altında.
- Üst çene (0,888) alt çeneden (0,848) biraz daha başarılı.

## Eğitim ayrıntıları

| | DGCNN | PointNet |
|---|---|---|
| Parametre | 1,02 M | 1,38 M |
| Girdi | 8.192 nokta × (X, Y, Z, Nx, Ny, Nz) | aynı |
| Epoch | 100 | 100 |
| En iyi doğrulama mIoU | 0,808 (epoch 86) | 0,717 (epoch 76) |
| Epoch süresi (A100) | ~32 sn | ~3 sn |

- Optimizasyon: AdamW (lr 0,001, weight decay 1e-4), cosine öğrenme oranı, bf16 karışık hassasiyet.
- Kayıp: sınıf ağırlıklı CrossEntropy (ENet tarzı ağırlıklar, w = 1/ln(1,02 + p)). Diş eti noktaların ~%45'ini oluşturduğu için gerekli.
- Veri artırma (sadece eğitimde): dikey eksen etrafında ±15° döndürme, ±%10 ölçekleme, Gauss gürültüsü.
- Çıkarım: mesh'in tüm köşeleri 8.192'lik rastgele parçalara bölünür. Her köşe 3 kez farklı komşulukla tahmin edilir, olasılıklar ortalanır (test-time voting).

## Eski model neden başarısızdı? (önemli bulgu)

Önceki sürümde mesh'ler `open3d.io.read_triangle_mesh` ile okunuyordu. Open3D, OBJ dosyasını okurken **köşelerin sırasını değiştiriyor**. Teeth3DS etiketleri ise OBJ'deki köşe sırasına göre verildiği için her diş etiketi çenenin rastgele köşelerine dağılıyordu. Model anlamlı bir şey öğrenemedi ve her noktaya en sık sınıf olan "diş eti" dedi:

- Nokta doğruluğu 0,459, yani neredeyse sadece diş eti oranı.
- Dişlerde mIoU 0.

Sırayı koruyan OBJ okuyucu yazıldı (`ai_model/dataset.py → obj_oku`) ve regresyon testi eklendi. Ayrıca şunlar da düzeltildi: doğrulama setine veri artırma uygulanması, hasta sızıntısı, sınıf dengesizliği, eğitim ile çıkarım arasındaki örnekleme farkı.

## Hata analizi

- **20 yaş dişleri en zayıf sınıflar** (DGCNN IoU: 18 → 0,65, 28 → 0,55, 38 → 0,52, 48 → 0,49). Bu dişler çoğu hastada yok, gömülü ya da kısmen sürmüş. Eğitimde az örnek var.
- **En sık karışan çiftler komşu azı dişleri** (karışıklık matrisi, satır normalize):

| Gerçek → Tahmin | Oran |
|---|---|
| 48 → 47 | %40,4 |
| 38 → 37 | %17,2 |
| 18 → diş eti | %8,8 |
| 18 → 17 | %6,8 |
| 27 → 26 | %6,1 |
| 17 → 16 | %6,1 |

  Hatalar neredeyse hiç karşı çeneye ya da karşı tarafa gitmiyor. Model dişin yerini doğru buluyor ama arka dişlerde **numarayı bir kaydırabiliyor**.
- En kötü vaka (016FSM14, mIoU 0,05): Diş sınırları doğru bulunmuş, ama eksik dişler yüzünden numaralandırma kaymış ve neredeyse tüm dişlere komşu numara verilmiş (Şekil 6). Bu, nokta bazlı modellerin bilinen bir zayıflığı. İleride diş sayısı ve sırası bilgisini kullanan bir son-işleme (ör. arkı boyunca sıralama) bunu düzeltebilir.

## Şekiller (`rapor_gorselleri/`)

| Dosya | İçerik |
|---|---|
| `sekil1_model_karsilastirma.png` | Üç modelin test metrikleri |
| `sekil2_ogrenme_egrileri.png` | Kayıp ve doğrulama mIoU eğrileri |
| `sekil3_dis_bazinda_iou.png` | 32 diş + diş eti için IoU, PointNet ile DGCNN |
| `sekil4_karisiklik_matrisi.png` | DGCNN 33×33 karışıklık matrisi |
| `sekil5_vaka_dagilimi.png` | Tarama başına mIoU dağılımı |
| `sekil6_ornek_tahminler.png` | En iyi ve en kötü vakalarda gerçek etiket ile tahmin |

## Kısıtlar

- Değerlendirme, veri setinin kendi etiketleriyle ve köşe bazında IoU/Dice üzerinden yapıldı. Teeth3DS yarışmasının resmî metrikleri (TSA, TLA, TIR) ve resmî test bölmesi kullanılmadı. Literatürle doğrudan sayısal karşılaştırma yapılırken bu not edilmeli.
- Tek tohum (seed = 42) ile tek eğitim yapıldı; varyans ölçülmedi.

---

# Parça Tarama Birleştirme Deneyi (data/four)

Betik: `experiments/birlestirme_deneyi.py`. Ham sonuçlar: `runs/birlestirme/` (A.json, B.json, C.json ve tablo_*.csv).

## Yöntem

- **Veri:** `data/four/partial_scans`. Her dosya tam çenenin bir dişi merkez alan kırpılmış parçası ve orijinal çene koordinatında duruyor, yani doğru konum (ground truth) biliniyor. 12 hasta, 127 parça, ark boyunca 115 komşu parça çifti. Komşu parçaların örtüşmesi %35–61.
- **Algoritmaya koordinat verilmedi.** Her parça:
  1. yüzeyinden bağımsız olarak yeniden örneklendi (~4 nokta/mm²; komşu parçalarda aynı noktalar kalmaz),
  2. üzerine Gauss sensör gürültüsü eklendi (σ = 0,02 mm),
  3. rastgele döndürülüp ötelenerek "konumu bilinmeyen tarama" hâline getirildi.

  Koordinatlar sadece değerlendirmede kullanıldı.
- **Hata ölçütü:** Parçanın noktalarının bulunan konum ile gerçek konum arasındaki sapması (RMSE, mm) ve dönme hatası (°). **Başarı eşiği: 0,1 mm.**
- **Hareket seviyeleri:**
  - küçük: 2–5° ve 1–3 mm (ardışık iki kare arasındaki gerçekçi prob hareketi)
  - orta: 10–15° ve 4–6 mm
  - büyük: 25–35° ve 8–12 mm (takip kaybı / yeniden konumlama)

## Deney A: Komşu parça çifti hizalama (115 çift, σ = 0,02 mm)

| Yöntem | Küçük hareket | Orta hareket | Büyük hareket | Başarılılarda ort. hata | Süre / çift |
|---|---|---|---|---|---|
| ICP (nokta-nokta) | %68 | %1 | %0 | 0,056 mm | 0,07 sn |
| ICP (nokta-düzlem) | %99 | %56 | %4 | 0,014 mm | 0,03–0,09 sn |
| RANSAC + ICP (her seferinde) | %98 | %97 | %94 | 0,014 mm | 0,21 sn |
| **Hibrit (ICP → gerekirse RANSAC)** | **%100** | **%100** | **%99** | **0,014 mm** | 0,11–0,32 sn |

- Nokta-düzlem ICP, nokta-nokta ICP'den çok daha iyi. Diş gibi pürüzsüz ve kavisli yüzeylerde beklenen bir sonuç.
- ICP tek başına sadece küçük harekette çalışıyor. Büyük harekette RANSAC (FPFH öznitelikleri ile kaba hizalama) şart.
- Başarılı hizalamaların doğruluğu yöntemden bağımsız olarak ~0,014 mm.

## Deney B: Tam ark birleştirme (12 hasta, 9–13 parça, küçük hareket)

| Strateji | Ort. hata | Medyan hata | En büyük hata | Son parçanın ort. hatası |
|---|---|---|---|---|
| Kare→kare (her seferinde RANSAC) | 1,56 mm | 0,065 mm | 6,9 mm | 2,37 mm |
| Kare→model (her seferinde RANSAC, ilk sürüm) | 1,6–2,2 mm* | 0,05 mm | 27–39 mm* | 1,2–1,4 mm* |
| **Kare→model (hibrit)** | **0,054 mm** | **0,038 mm** | **0,21 mm** | **0,11 mm** |

\* Her karede RANSAC kullanan stratejinin sonuçları, RANSAC'ın rastgeleliği nedeniyle çalıştırmadan çalıştırmaya değişiyor. Hibrit stratejinin sonuçları değişmiyor.

**Önemli bulgu:** İlk sürümdeki "her karede RANSAC" yaklaşımı, 12 hastanın 5'inde bazı parçaları yanlış yere oturttu (birbirine benzeyen komşu dişler). Bu tür bir hata kare→kare stratejisinde zincirin sonuna kadar taşındı. Hata analizinde, ICP'nin iç kalıntı hatasının (inlier RMSE) doğru hizalamalarda her zaman ≤ ~0,30 mm, yanlışlarda ise ≥ ~0,36 mm olduğu görüldü. Bu sayede algoritma kendi hatasını fark edebiliyor.

Buna dayanarak **hibrit mod** geliştirildi ve varsayılan yapıldı (`config.yaml → icp_stitcher.registration_mode: "hybrid"`):
1. Önce önceki konumdan ICP ile takip.
2. Sonuç kalite testini (örtüşme ≥ 0,45 ve inlier RMSE ≤ 0,55 × ICP mesafesi) geçemezse RANSAC ile yeniden konumlanma; farklı tohumla en fazla 3 deneme.

Hibrit modda hiçbir hastada büyük hata oluşmadı. Biriken kayma (drift) 11 parça boyunca ~0,01 mm'den ~0,1–0,2 mm'ye çıkıyor (Şekil 9). Bu kayma, gerçek sistemlerde döngü kapatma ve global optimizasyon (pose graph) ile azaltılır; gelecek çalışma olarak önerilir.

## Deney C: Gürültü duyarlılığı (hibrit yöntem, 115 çift)

| Gürültü σ | Küçük hareket başarı | Büyük hareket başarı | Medyan hata |
|---|---|---|---|
| 0 mm | %100 | %98 | 0,012 mm |
| 0,02 mm | %100 | %99 | 0,012 mm |
| 0,05 mm | %100 | %99 | 0,016 mm |
| 0,10 mm | %100 | %97 | 0,024 mm |
| 0,20 mm | %87 | %36 | 0,06–5,2 mm |

Yöntem 0,1 mm'ye kadar sensör gürültüsünde güvenilir. 0,2 mm'de özellikle büyük harekette (FPFH öznitelikleri bozulduğu için) çöküyor. Bu, tarayıcı donanımı için bir **tasarım gereksinimi** veriyor: nokta başına gürültü ≤ ~0,1 mm olmalı.

## Şekiller

| Dosya | İçerik |
|---|---|
| `sekil7_birlestirme_basari.png` | 4 yöntem × 3 hareket seviyesi başarı oranı |
| `sekil8_ortusme_hata.png` | Örtüşme oranı ile hata ilişkisi (hibrit) |
| `sekil9_drift.png` | Tam arkta parça sırasına göre biriken hata |
| `sekil10_sapma_haritasi.png` | Aynı hastada her karede RANSAC ile hibrit yöntemin sapma ısı haritası |
| `sekil11_gurultu.png` | Gürültü duyarlılığı |

## Kısıtlar

- Parçalar gerçek çene taramalarından kırpıldı. Gerçek bir tarayıcının ışık yansıması, eksik veri ve hareket bulanıklığı gibi hataları gürültü modeline dahil değil; sadece Gauss gürültüsü kullanıldı.
- Parçaların sırası (ardışık kareler) algoritmaya verildi. Gerçek tarayıcıda da kareler zaman sırasıyla geldiği için bu gerçekçi bir varsayım.

---

# Uçtan Uca Sanal Tarama Deneyi

Betikler: `simulation/sanal_tarayici.py`, `experiments/sanal_tarama_deneyi.py`, `experiments/sanal_tarama_duyarlilik.py`, `experiments/sanal_tarama_sekiller.py`. Ham sonuçlar: `runs/sanal_tarama/VBUD4E5R_lower/`.

## Yöntem

- **Referans model:** Teeth3DS test setinden bir alt çene (hasta VBUD4E5R, 138 bin köşe). Bu hasta segmentasyon modelinin eğitiminde hiç kullanılmadı. Seçimde en iyi vaka değil, test setinde **medyana yakın** bir vaka tercih edildi (orijinal modelde DGCNN mIoU 0,90).
- **Sanal tarayıcı (ışın izleme, Open3D RaycastingScene):**
  - Kamera: pinhole, 640×480, f = 600 px. Çalışma mesafesinde görüş alanı ~18×14 mm, piksel boyu ~0,03 mm.
  - Lazer: kamera ekseninden 8 mm ötede bir yayıcıdan çıkan kırmızı çizgi lazer tabakası. Optik ekseni 17 mm'de kesiyor (üçgenleme açısı ~25°).
  - Lazer çizgisi yüzeyde ~0,05 mm kalınlıkta. Gölgeler, yayıcıdan görünmeyen noktalar ve yüzey eğimine bağlı parlaklık hesaplanıyor.
  - Görüntü: koyu ortam ışığı, pembe diş eti (kırmızı lazerle karışabilen zor bir arka plan), Gauss kamera gürültüsü (σ = 3 / 255) ve hafif bulanıklık.
- **Tarama:** Prob ark boyunca üç geçiş yapıyor: tepeden (0°), yanaktan (+35°) ve dilden (−35°). Her 0,25 mm'de bir kare alınıyor; toplam **1.461 kare**.
- **Hat:** Projedeki gerçek kod kullanıldı: `LaserDetector` → `Triangulator` → birleştirme → istatistiksel aykırı nokta temizleme → Poisson yüzey (`MeshGenerator`) → DGCNN segmentasyonu.
  - Poisson için normaller, her noktanın görüldüğü kamera yönüne çevrildi.
  - Taranmayan bölgelerde Poisson'un uydurduğu yüzeyler kırpıldı (noktalara 0,4 mm'den uzak köşeler).
- **Ölçütler:**
  - Yeniden oluşturulan yüzeyin gerçek yüzeye uzaklığı (doğruluk; ISO 12836 "trueness" yaklaşımında en iyi rijit hizalamadan sonra).
  - Gerçek diş yüzeyinin yüzde kaçının 0,1 / 0,2 mm içinde yakalandığı (kapsama).
  - Yeniden oluşturulan model üzerinde segmentasyon mIoU'su (gerçek etiketler en yakın orijinal köşeden aktarıldı).

## Sonuçlar

| Senaryo | 3D nokta hatası (medyan / RMS) | Yüzey doğruluğu (medyan / RMS / %95) | Diş kapsaması (0,2 mm) | Segmentasyon mIoU |
|---|---|---|---|---|
| İdeal poz, eski dedektör | 0,006 / 0,019 mm (en büyük 1,69 mm) | 0,005 / 0,035 / 0,052 mm | %95,7 | — |
| **İdeal poz, yeni dedektör** | **0,006 / 0,016 mm (en büyük 0,40 mm)** | **0,005 / 0,035 / 0,053 mm** | **%95,1** | **0,941** |
| Sadece IMU (kaymalı, ort. poz hatası 1,06 mm) | 0,30 / 0,56 mm | 0,27 / 0,55 / 1,22 mm | %54,7 | 0,536 |
| **IMU + hibrit ICP + makullük kapısı** (ort. poz hatası 0,44 mm) | 0,17 / 0,28 mm | **0,11 / 0,23 / 0,50 mm** | %62,6 | **0,912** |

Ek bilgiler:
- 549 bin 3D nokta üretildi; yeniden oluşturulan yüzey ~180 bin köşeli.
- İdeal pozda diş eti kapsaması %64. Diş etinin dış kenarları çalışma alanı dışında kalıyor.
- **Yeniden oluşturulan model üzerinde segmentasyon (0,941), orijinal model üzerindeki sonuçtan (0,90) daha iyi çıktı.** Muhtemel sebep: Poisson yüzeyinin pürüzsüz olması ve diş etinin dış kenarlarının taranmamış olması. Tek bir vakadan genelleme yapılmamalı.

### Bulgu 1: Lazer dedektörü iyileştirildi

Sanal görüntülerde, aynı piksel sütununda iki ayrı lazer parçası olduğunda (ör. dişin önü ve kenarı) eski dedektörün ikisinin ağırlık merkezini alıp **aradaki boşluğa sahte nokta koyduğu** görüldü (Şekil 12). Dedektör artık her sütunda en güçlü bağlantılı parçayı seçiyor (`LaserDetector(single_peak=True)`, varsayılan):

- En büyük nokta hatası 1,69 mm'den **0,40 mm'ye** indi.
- RMS 0,019 mm'den **0,016 mm'ye** indi.
- Yüzey doğruluğu zaten aykırı nokta filtresi sayesinde etkilenmiyordu.

### Bulgu 2: Konum takibi

Tek lazer çizgisiyle her kare bir eğri olduğu için kareden kareye ICP mümkün değil (ilk analizdeki tespit). Bu yüzden:

1. Kısa vadeli hareket IMU'dan alındı. Simülasyonda gerçek konuma rastgele yürüyüş + sabit eğilim (2 µm/kare) şeklinde biriken kayma eklendi; kayma geçişler boyunca sıfırlanmıyor.
2. Kareler 40'lık yamalara (~10 mm) toplanıp her yama mevcut modele hibrit ICP ile hizalandı.

İlk denemede bazı yamalar birbirine benzeyen başka dişlere oturdu (ortalama poz hatası 30 mm). **Makullük kapısı** eklendi: IMU hatası sınırlı olduğu için 3 mm / 5°'den büyük düzeltmeler reddediliyor (`register(max_correction_mm, max_correction_deg)`). Sonuç:

| | Tepe geçişi | Yanak geçişi | Dil geçişi |
|---|---|---|---|
| Sadece IMU | 0,40 mm | 1,01 mm | 1,76 mm |
| IMU + ICP + kapı | 0,41 mm | 0,44 mm | 0,46 mm |

- Yanak ve dil geçişleri tepe geçişine kilitleniyor, kayma birikmiyor.
- Tepe geçişinin kendi kayması düzeltilemiyor, çünkü o geçişte yamalar arasında örtüşme yok. Gerçek cihazda bu ya daha iyi bir IMU ya da yapılandırılmış ışık / çoklu çizgi (her karede yüzey yaması) gerektirir.

### Duyarlılık analizi (Şekil 16; nokta hatası RMS, ideal değer 0,016 mm)

| Parametre | Hata | Nokta hatası RMS |
|---|---|---|
| Odak uzaklığı | %0,5 / %1 / %2 | 0,05 / 0,10 / 0,19 mm |
| **Lazer düzlemi açısı** | **0,25° / 0,5° / 1°** | **0,24 / 0,47 / 0,92 mm** |
| Kamera–lazer bazı | 0,05 / 0,1 / 0,2 mm | 0,09 / 0,18 / 0,36 mm |
| Kamera gürültüsü σ | 0 / 3 / 10 / 25 / 50 | 0,015 / 0,016 / 0,018 / 5,3* / 13* mm |
| IMU kayma eğilimi (µm/kare) | 0 / 1 / 2 / 5 / 10 | ICP ile 0,37 / 0,29 / 0,44 / 2,98 / 6,77 mm poz hatası (sadece IMU: 0,75 / 0,55 / 1,06 / 3,14 / 6,77) |

\* σ ≥ 25'te gürültü pikselleri lazer olarak algılanıyor (aykırı noktalar). Medyan hata σ = 25'te hâlâ 0,02 mm.

**Tasarım gereksinimleri (çıkarım):**
- 0,05 mm doğruluk için lazer düzlemi açısı ~0,05°, baz uzunluğu ~0,02 mm ve odak uzaklığı ~%0,5 hassasiyetle kalibre edilmeli. **En kritik parametre lazer düzlemi açısı.**
- ICP düzeltmesi, IMU kayması kapı sınırının (3 mm) altında kaldığı sürece işe yarıyor. Daha kötü bir IMU için kapı genişletilmeli; bu da yanlış eşleşme riskini artırır.

### Ek bulgu: Piksel kilitlenmesi

Düz yüzey testinde (`tests/test_sanal_tarayici.py`) ağırlık merkezi yönteminin, çizginin piksel ızgarasına göre konumuna bağlı olarak ~0,02 mm'ye kadar sistematik sapma yaptığı görüldü ("pixel locking"). Gauss eğrisi uydurma gibi yöntemler bunu azaltabilir (gelecek çalışma).

## Şekiller

| Dosya | İçerik |
|---|---|
| `sekil12_sanal_kamera_tespit.png` | Sanal kamera görüntüsü; eski ve yeni dedektör (büyütülmüş) |
| `sekil13_sanal_tarama_sapma_kapsama.png` | Yeniden oluşturulan yüzeyin sapma haritası ve kapsama haritası |
| `sekil14_sanal_tarama_segmentasyon.png` | Yeniden oluşturulan model üzerinde gerçek etiket ile DGCNN tahmini |
| `sekil15_uctan_uca.png` | Uçtan uca akış: görüntü → nokta bulutu → yüzey → segmentasyon |
| `sekil16_duyarlilik.png` | Kalibrasyon, IMU ve gürültü duyarlılığı |
| `sekil17_poz_senaryolari.png` | İdeal poz / sadece IMU / IMU + ICP karşılaştırması (sapma ve segmentasyon) |

## Kısıtlar

- Tek bir hasta (alt çene) üzerinde yapıldı. Üst çene ve daha fazla hasta ile tekrarlanmalı.
- Işık modeli basit: Lambert yansıması ve Gauss çizgi profili. Mine içinde lazer saçılması (subsurface scattering), parlak yansımalar, tükürük ve hareket bulanıklığı modellenmedi. Gerçek cihazda doğruluk bu değerlerden kötü olacaktır; buradaki sayılar yöntemin **üst sınırını** gösterir.
- IMU modeli basit (rastgele yürüyüş + sabit eğilim). Gerçek IMU'larda sıcaklıkla değişen eğilim ve ivmeölçer çift integrasyonu hataları daha büyüktür.

---

# 4. Otomatik klinik ölçümler

Segmentasyonun klinikte ne işe yaradığını göstermek için, ortodontistin elle yaptığı model analizinin temel ölçümleri dişlerin etiketlerinden otomatik çıkarıldı (`processing/klinik_olcum.py`).

## Yöntem

- **Okluzal düzlem:** Diş merkezlerine düzlem oturtulur. x ekseni hastanın sağından soluna, y ekseni arkadan öne, n ekseni diş tepelerine doğru alınır.
- **Etiket temizliği:** Her etiketin yalnızca en büyük bağlı parçası tutulur. 12 mm²'den küçük parçalar "diş yok" sayılır.
- **Meziodistal genişlik:** Kuron noktaları ark yönüne izdüşürülür. Ark yönü, diş merkezlerinden geçen parametrik 4. derece eğrinin teğetidir; arkın sonundaki dişte komşu dişin merkezine doğru alınır. Genişlik, izdüşümün %2–%98 aralığıdır; sınırdaki yanlış etiketlere karşı bu kırpma yapılır ve kırpılan pay geri eklenir.
- **Mevcut ark uzunluğu:** 1–6 numaralı dişlerin merkezlerinden geçen düzgün ark eğrisinin, iki taraftaki 6 numaralı dişin meziyal kenarı arasındaki uzunluğu.
- **Gerekli alan:** İki taraftaki 1–5 numaralı dişlerin (10 diş) genişlikleri toplamı.
- **Ark uzunluğu farkı:** Gerekli alan − mevcut ark. Pozitif değer çapraşıklık, negatif değer boşluk demektir.
- **Diğer ölçümler:** İnterkanin genişlik (iki köpek dişinin tepe noktaları arası), intermolar genişlik (iki 6 numaralı dişin okluzal merkezleri arası), Bolton ön (3–3) ve toplam (6–6) oranları.
- **Parametrelerin seçimi:** Kırpma oranı ve ark sonu yönü, test setinden ayrı olarak doğrulama bölümünden alınan 14 taramada seçildi (`runs/klinik_olcum_ayar`).

**Doğrulamanın anlamı:** Aynı algoritma uzmanın etiketlediği segmentasyona da uygulanır, çıkan değer referans kabul edilir. Fark, *segmentasyon hatasının ölçüme etkisini* gösterir. Ölçüm tanımının kendisi (örneğin kumpasla ölçülen temas noktası genişliği ile otomatik genişliğin farkı) bu veri setiyle doğrulanamaz, çünkü veri setinde elle yapılmış ölçüm yok.

## Sonuçlar: test seti (31 tarama, 418 diş, DGCNN tek tur çıkarım)

| Ölçüm | Ortalama mutlak hata | Ortalama fark (yapay zekâ − uzman) |
|---|---|---|
| Diş genişliği (tüm dişler) | **0,34 mm** (medyan 0,09 mm; dişlerin %86'sı 0,5 mm içinde) | +0,19 mm |
| Kesici ve köpek dişleri | 0,16–0,18 mm | |
| Küçük azılar | 0,19–0,22 mm | |
| 1. büyük azı | 0,58 mm | |
| 2. büyük azı | 0,98 mm | |
| Mevcut ark uzunluğu | 1,26 mm | +1,09 mm |
| Gerekli alan | 1,17 mm | +1,13 mm |
| **Ark uzunluğu farkı (çapraşıklık)** | **1,25 mm** | +0,33 mm |
| İnterkanin genişlik | 0,43 mm | |
| İntermolar genişlik | 0,45 mm | |
| Çapraşıklık sınıfı (yok / hafif ≤4 / orta 4–8 / şiddetli >8 mm) | Uyum %78 (27 taramanın 21'i) | |

Model 5 dişi bulamadı, 4 fazla diş buldu. Büyük hatalar arkın sonundaki büyük azılarda toplanıyor; segmentasyonun en zayıf olduğu dişler de bunlar. Ön bölgede hata 0,2 mm'nin altında.

İlk sürümde kırpma %0,5 ve ark sonu yönü eğriden alınıyordu. O sürümde genişlik hatası 0,44 mm, çapraşıklık hatası 1,80 mm idi. Doğrulama setinde seçilen ayarlarla bu değerler 0,34 mm ve 1,25 mm'ye indi.

## Sanal taramadan elde edilen modellerde (VBUD4E5R, alt ve üst çene)

| Ayar | Diş genişliği hatası (alt / üst) | Ortalama genişlik farkı | Çapraşıklık hatası (alt / üst) |
|---|---|---|---|
| 0,5 mm, konum biliniyor | 0,22 / 0,27 mm | −0,21 / −0,25 mm | −1,9 / −2,8 mm |
| 0,5 mm, IMU + ICP | 0,28 / 0,42 mm | −0,26 / −0,17 mm | −1,9 / −3,8 mm |
| 0,5 mm, sadece IMU | 0,81 / 1,93 mm | −0,43 / −0,42 mm | −3,4 / +2,6 mm |
| 1 mm, konum biliniyor | 0,47 / 0,60 mm | −0,47 / −0,36 mm | −4,3 / −3,9 mm |

**Bulgu:** Taranan modelde dişler, iyi bir taramada bile, sistematik olarak 0,2–0,25 mm dar ölçülüyor. Sebebi, lazer çizgisinin iki dişin temas ettiği arayüz bölgesini görememesi: komşu diş gölge yapıyor ve bu bölgede yüzey eksik kalıyor. Bu küçük fark 10 dişte toplanınca çapraşıklık yaklaşık 2–3 mm eksik hesaplanıyor. Gerçek ağız içi tarayıcılarda da bilinen bir zorluk bu; ek açılardan geçiş ya da temas bölgesine yönelik tarama ile azaltılabilir. Kareler arası mesafe 1 mm'ye çıkınca ve konum takibi bozulunca hata belirgin şekilde artıyor.

## Şekil

- `sekil18_klinik_olcum.png`: (a) Yapay zekâ ve uzman etiketlerinden ölçülen diş genişlikleri; (b) Bland–Altman grafiği; (c) ark uzunluğu farkı, test seti ve sanal taramalar birlikte.

## Kısıtlar

- Referans, uzman segmentasyonuna uygulanan aynı algoritmadır; elle (kumpasla) yapılmış bir ölçümle karşılaştırma yapılamadı.
- Veri setinde ısırma kaydı olmadığı için Bolton oranları gösterilebiliyor, ama kapanış ilişkisine dayanan ölçümler (overjet, overbite) yapılamıyor.
- Çok çapraşık vakalarda (örneğin dil tarafına kaçmış kesiciler, 016FSM14) ark eğrisi bu dişleri de içerdiği için mevcut ark kısa hesaplanabiliyor. Eksik dişi olan vakalarda gerekli alan hesaplanmıyor; arayüzde uyarı gösteriliyor.
