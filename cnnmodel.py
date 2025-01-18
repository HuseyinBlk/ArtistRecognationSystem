import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
from tensorflow.keras.layers import Dropout
from PIL import Image


# 1. Veri Seti Hazırlığı
data_dir = r"C:\Users\Administrator\Desktop\dataset2"  # Veri seti klasörü
spectrogram_dir = "spectrograms"  # Kaydedilecek mel-spektrogramlar


def add_noise(data, noise_factor=0.1):
    """Ses verisine gürültü ekler."""
    noise = noise_factor * np.random.randn(len(data))
    augmented_data = data + noise
    return np.clip(augmented_data, -1.0, 1.0)

def adjust_pitch(data, sr, n_steps=2):
    """Ses verisinin pitch'ini değiştirir."""
    return librosa.effects.pitch_shift(data, sr=sr, n_steps=n_steps)

def adjust_speed(data, speed_factor=1.2):
    """Ses hızını değiştirir."""
    return librosa.effects.time_stretch(data, rate=speed_factor)

def create_spectrogram(file_path, save_path, augmentations=None):
    """Ses dosyasından mel-spektrogram çıkar ve kaydet."""
    if not os.path.exists(save_path):  # Eğer spektrogram daha önce oluşturulmamışsa
        y, sr = librosa.load(file_path, duration=10)

        # Uygulanacak augmentations listesi
        if augmentations:
            for augment in augmentations:
                if callable(augment):
                    if 'sr' in augment.__code__.co_varnames:
                        y = augment(y, sr)
                    else:
                        y = augment(y)

        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        plt.figure(figsize=(2.56, 2.56))  # 128x128 pixel görüntü
        librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max),
                                 y_axis='mel', fmax=8000, x_axis='time')
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

# Şarkıcı etiketlerini tanımla

os.makedirs(spectrogram_dir, exist_ok=True)

augmentations = {
    "normal": [],
    "noise": [add_noise], #arka plan gürültü
    "pitch": [lambda y, sr: adjust_pitch(y, sr, n_steps=2)], #ses tonu
    "speed": [lambda y: adjust_speed(y, speed_factor=1.2)] #ses hızı
}

data = [] #mel spektrogram dosya yolu
singer_labels_list = [] #mel spektrogram hangi sanatçıya ait
singer_labels = {singer: idx for idx, singer in enumerate(os.listdir(data_dir))} #şarkıcı etiketi

# Tüm şarkılar için mel-spektrogram oluştur
for singer in tqdm(os.listdir(data_dir), desc="Processing Singers"): #her bir şarkıcı dosyasını gezme
    singer_label = singer_labels[singer] #oluşturulan etiketler ile şarkıcı isimlerini eşleme
    singer_dir = os.path.join(data_dir, singer)
    for file in tqdm(os.listdir(singer_dir), desc=f"Processing {singer}", leave=False): #şarkıcı klasöründeki şarkıları listeler
        file_path = os.path.join(singer_dir, file) #şarkıya ait dosya yolu oluşturur
        for aug_type, aug_list in augmentations.items():
            spectrogram_path = os.path.join(spectrogram_dir, f"{singer}_{file}_{aug_type}.png")
            create_spectrogram(file_path, spectrogram_path, augmentations=aug_list)
            data.append(spectrogram_path)
            singer_labels_list.append(singer_label)

# 2. Veriyi Bölme
X_train, X_test, y_train, y_test = train_test_split(
    data, singer_labels_list, test_size=0.2, random_state=42
)

# Görüntüleri yükleme ve normalizasyon
def load_image(image_path):
    """Mel-spektrogram görüntüsünü yükler ve normalize eder."""

    img = Image.open(image_path).convert('L')  # Gri tonlamaya çevir
    img = img.resize((128, 128))  # 128x128 boyutuna çevir
    return np.array(img) / 255.0

X_train = np.array([load_image(path) for path in X_train]).reshape(-1, 128, 128, 1)
X_test = np.array([load_image(path) for path in X_test]).reshape(-1, 128, 128, 1)
y_train = to_categorical(y_train, num_classes=len(singer_labels))
y_test = to_categorical(y_test, num_classes=len(singer_labels))

# 3. CNN Modeli
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(singer_labels), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Modeli Eğitme
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50, batch_size=32
)

# 5. Modeli Değerlendirme
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Modeli Kaydet
model.save("cnnmodelwithaugmentations1.h5")

# 6. Tahmin
def predict_singer(file_path):
    """Bir ses dosyasının orijinal şarkıcısını tahmin eder."""
    spectrogram_path = "temp_spectrogram.png"
    create_spectrogram(file_path, spectrogram_path)  # Sadece eksikse oluştur
    img = load_image(spectrogram_path).reshape(1, 128, 128, 1)
    prediction = model.predict(img)
    predicted_label = np.argmax(prediction)
    return list(singer_labels.keys())[predicted_label]

# Örnek Tahmin
example_file = r"C:\Users\Administrator\Desktop\dataset2\Adele\62_14.wav"  # Tahmin etmek istediğiniz ses dosyası
predicted_singer = predict_singer(example_file)
print(f"Predicted Singer: {predicted_singer}")
"""
Epoch 1/50
517/517 [==============================] - 63s 122ms/step - loss: 2.2390 - accuracy: 0.1719 - val_loss: 2.1233 - val_accuracy: 0.2435
Epoch 2/50
517/517 [==============================] - 62s 120ms/step - loss: 1.8975 - accuracy: 0.3201 - val_loss: 1.4993 - val_accuracy: 0.4930
Epoch 3/50
517/517 [==============================] - 62s 120ms/step - loss: 1.5252 - accuracy: 0.4640 - val_loss: 1.2581 - val_accuracy: 0.5912
Epoch 4/50
517/517 [==============================] - 62s 120ms/step - loss: 1.3223 - accuracy: 0.5351 - val_loss: 1.0897 - val_accuracy: 0.6439
Epoch 5/50
517/517 [==============================] - 63s 122ms/step - loss: 1.1624 - accuracy: 0.5900 - val_loss: 0.9875 - val_accuracy: 0.6838
Epoch 6/50
517/517 [==============================] - 63s 122ms/step - loss: 1.0296 - accuracy: 0.6328 - val_loss: 0.8260 - val_accuracy: 0.7297
Epoch 7/50
517/517 [==============================] - 62s 119ms/step - loss: 0.9144 - accuracy: 0.6704 - val_loss: 0.7263 - val_accuracy: 0.7703
Epoch 8/50
517/517 [==============================] - 62s 120ms/step - loss: 0.8184 - accuracy: 0.7106 - val_loss: 0.6856 - val_accuracy: 0.7751
Epoch 9/50
517/517 [==============================] - 61s 118ms/step - loss: 0.7428 - accuracy: 0.7338 - val_loss: 0.6375 - val_accuracy: 0.7959
Epoch 10/50
517/517 [==============================] - 61s 119ms/step - loss: 0.6703 - accuracy: 0.7578 - val_loss: 0.6101 - val_accuracy: 0.8000
Epoch 11/50
517/517 [==============================] - 62s 119ms/step - loss: 0.6229 - accuracy: 0.7712 - val_loss: 0.6064 - val_accuracy: 0.8097
Epoch 12/50
517/517 [==============================] - 61s 119ms/step - loss: 0.5884 - accuracy: 0.7812 - val_loss: 0.5550 - val_accuracy: 0.8182
Epoch 13/50
517/517 [==============================] - 61s 119ms/step - loss: 0.5640 - accuracy: 0.7924 - val_loss: 0.5525 - val_accuracy: 0.8334
Epoch 14/50
517/517 [==============================] - 61s 118ms/step - loss: 0.5130 - accuracy: 0.8062 - val_loss: 0.5447 - val_accuracy: 0.8303
Epoch 15/50
517/517 [==============================] - 61s 119ms/step - loss: 0.4841 - accuracy: 0.8141 - val_loss: 0.5339 - val_accuracy: 0.8387
Epoch 16/50
517/517 [==============================] - 61s 119ms/step - loss: 0.4531 - accuracy: 0.8284 - val_loss: 0.5383 - val_accuracy: 0.8428
Epoch 17/50
517/517 [==============================] - 61s 119ms/step - loss: 0.4327 - accuracy: 0.8369 - val_loss: 0.5182 - val_accuracy: 0.8496
Epoch 18/50
517/517 [==============================] - 61s 118ms/step - loss: 0.4278 - accuracy: 0.8367 - val_loss: 0.4970 - val_accuracy: 0.8489
Epoch 19/50
517/517 [==============================] - 61s 118ms/step - loss: 0.4159 - accuracy: 0.8389 - val_loss: 0.5051 - val_accuracy: 0.8474
Epoch 20/50
517/517 [==============================] - 61s 118ms/step - loss: 0.3980 - accuracy: 0.8495 - val_loss: 0.5627 - val_accuracy: 0.8402
Epoch 21/50
517/517 [==============================] - 62s 119ms/step - loss: 0.3757 - accuracy: 0.8589 - val_loss: 0.5355 - val_accuracy: 0.8443
Epoch 22/50
517/517 [==============================] - 61s 118ms/step - loss: 0.3638 - accuracy: 0.8590 - val_loss: 0.5257 - val_accuracy: 0.8561
Epoch 23/50
517/517 [==============================] - 61s 119ms/step - loss: 0.3612 - accuracy: 0.8597 - val_loss: 0.5206 - val_accuracy: 0.8590
Epoch 24/50
517/517 [==============================] - 65s 126ms/step - loss: 0.3614 - accuracy: 0.8614 - val_loss: 0.5805 - val_accuracy: 0.8433
Epoch 25/50
517/517 [==============================] - 64s 123ms/step - loss: 0.3460 - accuracy: 0.8675 - val_loss: 0.5634 - val_accuracy: 0.8501
Epoch 26/50
517/517 [==============================] - 64s 123ms/step - loss: 0.3415 - accuracy: 0.8674 - val_loss: 0.5254 - val_accuracy: 0.8578
Epoch 27/50
517/517 [==============================] - 62s 121ms/step - loss: 0.3184 - accuracy: 0.8779 - val_loss: 0.5615 - val_accuracy: 0.8574
Epoch 28/50
517/517 [==============================] - 61s 119ms/step - loss: 0.3279 - accuracy: 0.8718 - val_loss: 0.5713 - val_accuracy: 0.8564
Epoch 29/50
517/517 [==============================] - 62s 119ms/step - loss: 0.3241 - accuracy: 0.8743 - val_loss: 0.5716 - val_accuracy: 0.8537
Epoch 30/50
517/517 [==============================] - 61s 119ms/step - loss: 0.3133 - accuracy: 0.8776 - val_loss: 0.5683 - val_accuracy: 0.8605
Epoch 31/50
517/517 [==============================] - 62s 119ms/step - loss: 0.2949 - accuracy: 0.8832 - val_loss: 0.6019 - val_accuracy: 0.8610
Epoch 32/50
517/517 [==============================] - 62s 119ms/step - loss: 0.2862 - accuracy: 0.8877 - val_loss: 0.6003 - val_accuracy: 0.8581
Epoch 33/50
517/517 [==============================] - 61s 118ms/step - loss: 0.2981 - accuracy: 0.8848 - val_loss: 0.5788 - val_accuracy: 0.8622
Epoch 34/50
517/517 [==============================] - 61s 118ms/step - loss: 0.2879 - accuracy: 0.8932 - val_loss: 0.6200 - val_accuracy: 0.8566
Epoch 35/50
517/517 [==============================] - 61s 118ms/step - loss: 0.2736 - accuracy: 0.8958 - val_loss: 0.5735 - val_accuracy: 0.8651
Epoch 36/50
517/517 [==============================] - 61s 118ms/step - loss: 0.2686 - accuracy: 0.8990 - val_loss: 0.6059 - val_accuracy: 0.8644
Epoch 37/50
517/517 [==============================] - 60s 117ms/step - loss: 0.2660 - accuracy: 0.8975 - val_loss: 0.6767 - val_accuracy: 0.8617
Epoch 38/50
517/517 [==============================] - 61s 118ms/step - loss: 0.2648 - accuracy: 0.9001 - val_loss: 0.5772 - val_accuracy: 0.8588
Epoch 39/50
517/517 [==============================] - 61s 119ms/step - loss: 0.2693 - accuracy: 0.8946 - val_loss: 0.6518 - val_accuracy: 0.8583
Epoch 40/50
517/517 [==============================] - 61s 118ms/step - loss: 0.2649 - accuracy: 0.8985 - val_loss: 0.5684 - val_accuracy: 0.8673
Epoch 41/50
517/517 [==============================] - 61s 119ms/step - loss: 0.2542 - accuracy: 0.9013 - val_loss: 0.6010 - val_accuracy: 0.8646
Epoch 42/50
517/517 [==============================] - 62s 119ms/step - loss: 0.2541 - accuracy: 0.9007 - val_loss: 0.5857 - val_accuracy: 0.8704
Epoch 43/50
517/517 [==============================] - 61s 118ms/step - loss: 0.2474 - accuracy: 0.9057 - val_loss: 0.6271 - val_accuracy: 0.8675
Epoch 44/50
517/517 [==============================] - 61s 118ms/step - loss: 0.2513 - accuracy: 0.9048 - val_loss: 0.6260 - val_accuracy: 0.8615
Epoch 45/50
517/517 [==============================] - 61s 118ms/step - loss: 0.2386 - accuracy: 0.9091 - val_loss: 0.5686 - val_accuracy: 0.8697
Epoch 46/50
517/517 [==============================] - 61s 117ms/step - loss: 0.2375 - accuracy: 0.9103 - val_loss: 0.5876 - val_accuracy: 0.8743
Epoch 47/50
517/517 [==============================] - 60s 117ms/step - loss: 0.2334 - accuracy: 0.9123 - val_loss: 0.6131 - val_accuracy: 0.8675
Epoch 48/50
517/517 [==============================] - 61s 118ms/step - loss: 0.2487 - accuracy: 0.9038 - val_loss: 0.6495 - val_accuracy: 0.8651
Epoch 49/50
517/517 [==============================] - 61s 118ms/step - loss: 0.2253 - accuracy: 0.9144 - val_loss: 0.6193 - val_accuracy: 0.8711
Epoch 50/50
517/517 [==============================] - 61s 119ms/step - loss: 0.2213 - accuracy: 0.9164 - val_loss: 0.6193 - val_accuracy: 0.8692
130/130 [==============================] - 4s 29ms/step - loss: 0.6193 - accuracy: 0.8692
Test Accuracy: 0.87
"""