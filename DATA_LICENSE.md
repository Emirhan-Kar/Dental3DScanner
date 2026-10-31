# Veri Lisansı / Data License

## Teeth3DS veri seti

Bu projedeki segmentasyon modeli **Teeth3DS** (3DTeethSeg'22 MICCAI Challenge) veri seti ile eğitilmiş ve değerlendirilmiştir. Veri seti bu projenin parçası değildir ve bu depoda **yer almaz**. Veri seti [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/) lisansıyla, yalnızca ticari olmayan araştırma ve eğitim amaçları için sunulmaktadır. Ticari kullanım için veri sahiplerinden ayrı lisans alınması gerekir.

- Lisansın tam metni (veri setiyle birlikte dağıtılan dosya): [lisanslar/Teeth3DS_CC-BY-NC-ND-4.0.txt](lisanslar/Teeth3DS_CC-BY-NC-ND-4.0.txt)
- Kaynak ve erişim: https://github.com/abenhamadou/3DTeethSeg22_challenge
- Veriyi indirdikten sonra `data/3DTeethData/data_part_1 … data_part_7` klasörlerine yerleştirin.

Veri setini kullanan çalışmalar aşağıdaki iki yayına atıf yapmalıdır (tam yazar listesi için kaynak sayfadaki "Citing us" bölümüne bakınız):

```bibtex
@article{ben2022teeth3ds,
  title   = {Teeth3DS: a benchmark for teeth segmentation and labeling from intra-oral 3D scans},
  author  = {Ben-Hamadou, Achraf and others},
  journal = {arXiv preprint arXiv:2210.06094},
  year    = {2022}
}

@article{ben2023_3dteethseg22,
  title   = {3DTeethSeg'22: 3D Teeth Scan Segmentation and Labeling Challenge},
  author  = {Ben-Hamadou, Achraf and others},
  journal = {arXiv preprint arXiv:2305.18277},
  year    = {2023}
}
```

## Veri setinden türetilen dosyalar

CC BY-NC-ND 4.0 lisansının "türev eser yok" koşulu nedeniyle, veri setindeki çene modellerinden doğrudan türetilen dosyalar bu depoya **konmamıştır** (bkz. `.gitignore`). Bu dosyalar kodla yerel olarak yeniden üretilebilir:

- Yeniden oluşturulmuş ya da renklendirilmiş mesh/nokta bulutları (`*.ply`, `*.obj`, `*.stl`)
- Sanal tarayıcının ürettiği kamera görüntüleri ve demo kayıtları (`runs/demo/`, `runs/sanal_tarama/**/*.npz`)

Depoda yalnızca toplu sonuçlar (metrik tabloları, istatistik dosyaları, grafikler) ve tarama başına tahmin edilen diş etiketleri (`runs/klinik_olcum*/tahminler/*.npy`, mesh içermez) bulunur. Rapor şekillerindeki çene görselleri, bilimsel raporlama amacıyla ve yukarıdaki atıfla kullanılmıştır.

## Eğitilmiş model ağırlıkları

`ai_model/best_dental_model.pth` Teeth3DS ile eğitilmiştir. Ağırlıklar proje lisansı ([LICENSE](LICENSE), yalnızca akademik kullanım) ile paylaşılır. Ayrıca veri setinin ticari olmayan kullanım koşulu da bu ağırlıklar için geçerli kabul edilmelidir. Ağırlıklar **klinik kullanım için değildir**.

---

*English summary:* The Teeth3DS dataset is third-party data licensed under CC BY-NC-ND 4.0 (non-commercial research and education only). It is not included in this repository. Files derived directly from its meshes are not redistributed and can be regenerated locally with the code. The model weights were trained on Teeth3DS, are shared under the project's academic-only license, are additionally subject to the dataset's non-commercial terms, and are not for clinical use.
