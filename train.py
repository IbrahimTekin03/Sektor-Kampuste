import tensorflow as tf
import os
import sys
from src.config import *

# Keras modüllerini doğrudan tf üzerinden çağırıyoruz (Hata riskini sıfırlar)
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
MobileNetV2 = tf.keras.applications.MobileNetV2
layers = tf.keras.layers
models = tf.keras.models
optimizers = tf.keras.optimizers
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint
EarlyStopping = tf.keras.callbacks.EarlyStopping


def setup_device():
    """Cihaz ve GPU yapılandırması"""
    print("🔍 Donanım taranıyor...")
    # TensorFlow sürümünü de yazdıralım ki emin olalım
    print(f"TensorFlow Sürümü: {tf.__version__}")

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"🚀 BAŞARILI: GPU Aktif -> {gpus[0].name}")
        except RuntimeError as e:
            print(f"GPU Hatası: {e}")
    else:
        print("⚠️ GPU Bulunamadı. İşlemci (CPU) kullanılacak.")


def train():
    setup_device()

    # 1. Veri Hazırlığı
    print(f"📂 Veri seti yolu kontrol ediliyor: {DATASET_PATH}")

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=30,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    try:
        train_generator = train_datagen.flow_from_directory(
            DATASET_PATH,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='training'
        )

        validation_generator = train_datagen.flow_from_directory(
            DATASET_PATH,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation'
        )
    except FileNotFoundError:
        print("❌ HATA: Veri seti klasörü bulunamadı!")
        print(f"Lütfen '{DATASET_PATH}' klasörünün dolu olduğundan emin olun.")
        return

    # 2. Model Mimarisi
    base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 3. Callbacks
    if not os.path.exists('models'):
        os.makedirs('models')

    callbacks_list = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True)
    ]

    # 4. EĞİTİM
    print(f"Eğitim Başlıyor... ({EPOCHS} Epoch)")

    try:
        model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=validation_generator,
            callbacks=callbacks_list
        )
    except Exception as e:
        print(f"Eğitim sırasında bir hata oluştu: {e}")
        return

    # Sınıf İsimlerini Kaydet
    class_names = list(train_generator.class_indices.keys())
    import pickle
    with open('models/class_names.pkl', 'wb') as f:
        pickle.dump(class_names, f)

    print(f"İşlem Tamam! Model kaydedildi: {MODEL_PATH}")


if __name__ == "__main__":
    if not os.path.exists(DATASET_PATH):
        print(f"HATA: '{DATASET_PATH}' klasörü yok!")
        print("Lütfen veri setini indirdiğinden ve isminin doğru olduğundan emin ol.")
    else:
        train()