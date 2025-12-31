# 🌿Photo Scan AI: Yapay Zeka Destekli Hayvan Sınıflandırma

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-green)

Photo Scan AI, derin öğrenme (Deep Learning) tekniklerini kullanarak **Animals-10** veri setindeki canlı türlerini yüksek doğrulukla sınıflandıran, kullanıcı dostu bir görüntü analiz sistemidir.

Proje, **Transfer Learning (MobileNetV2)** mimarisi üzerine kurulmuş olup, GPU hızlandırması ve modern bir web arayüzü sunar.

## 🚀 Özellikler

* **Transfer Learning:** ImageNet ağırlıkları ile eğitilmiş MobileNetV2 mimarisi.
* **Yüksek Doğruluk:** 10 farklı hayvan sınıfında optimize edilmiş sonuçlar.
* **Modern Arayüz:** Streamlit ile geliştirilmiş, Glassmorphism tasarım diline sahip responsive arayüz.
* **GPU Desteği:** Sistemde NVIDIA GPU varsa otomatik algılar ve eğitimi hızlandırır.
* **Görsel Analiz:** Tahmin sonuçlarını detaylı olasılık grafikleriyle (Bar Chart) sunar.

## 📂 Desteklenen Sınıflar (Animals-10)

Bu model aşağıdaki 10 sınıfı tanımak üzere eğitilmiştir:
`Köpek`, `Kedi`, `At`, `Koyun`, `İnek`, `Fil`, `Kelebek`, `Tavuk`, `Örümcek`, `Sincap`.

## 🛠️ Kurulum

Projeyi yerel makinenizde çalıştırmak için adımları takip edin.

### 1. Gerekli Kütüphaneleri Yükleyin
Terminali proje klasöründe açın ve bağımlılıkları yükleyin:

```bash
pip install -r requirements.txt
```

### Veri Setini Hazırlayın
Bu proje Animals-10 veri setini kullanır. Veri seti boyutu nedeniyle repoya dahil edilmemiştir.
1. Kaggle Animals-10 adresinden veri setini indirin.
2. İndirdiğiniz zip dosyasını çıkarın.
3. raw-img klasörünü projenin içindeki dataset klasörüne taşıyın.

### Kullanım
Modeli Eğitmek İçin sıfırdan eğitim başlatmak için aşağıdaki komutu çalıştırın. Kod, GPU varsa otomatik kullanacaktır.
```bash
python train.py
```

### Arayüzü Başlatmak İçin
Web arayüzünü açmak için:
```bash
streamlit run app.py
```
