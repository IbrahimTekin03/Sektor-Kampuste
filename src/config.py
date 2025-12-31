import os

# Resim Ayarları (MobileNetV2 standardı 224x224'tür, en iyi sonucu verir)
IMG_SIZE = (224, 224)
BATCH_SIZE = 32  # GPU belleğin düşükse (örn: 4GB altı) bunu 16 yap
EPOCHS = 10      # 10 Tur eğitim

# Dosya Yolları
# İndirdiğin veri setinin içinde 'raw-img' klasörü var, yolu ona göre veriyoruz
DATASET_PATH = os.path.join("dataset", "raw-img") 
MODEL_PATH = os.path.join("models", "animals10_gpu_model.h5")

# Sınıflar (Animals-10 İtalyanca klasör isimleriyle gelir, bunları Türkçeye çevireceğiz)
# Klasör isimleri: cane, cavallo, elefante, farfalla, gallina, gatto, mucca, pecora, ragno, scoiattolo
CLASS_TRANSLATIONS = {
    'cane': 'Köpek',
    'cavallo': 'At',
    'elefante': 'Fil',
    'farfalla': 'Kelebek',
    'gallina': 'Tavuk',
    'gatto': 'Kedi',
    'mucca': 'İnek',
    'pecora': 'Koyun',
    'ragno': 'Örümcek',
    'scoiattolo': 'Sincap'
}