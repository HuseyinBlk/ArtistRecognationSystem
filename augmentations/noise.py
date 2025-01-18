import numpy as np
import librosa
import sounddevice as sd

def add_noise(data, noise_factor=0.1):
    """Ses verisine gürültü ekler."""
    noise = noise_factor * np.random.randn(len(data))
    augmented_data = data + noise
    return np.clip(augmented_data, -1.0, 1.0)  # Veriyi -1 ile 1 arasında sınırlıyoruz

# Ses dosyasını yükle
file_path = r"C:\Users\Administrator\Desktop\dataset2\Adele\62_14.wav"  # Ses dosyanızın yolu
data, sr = librosa.load(file_path, sr=None)  # sr=None ile orijinal örnekleme hızını korur

# Gürültü eklenmiş veriyi oluştur
noisy_data = add_noise(data, noise_factor=0.05)

# Gürültü eklenmiş sesi çal
print("Gürültü eklenmiş sesi çalıyor...")
sd.play(noisy_data, samplerate=sr)
sd.wait()  # Çalma tamamlanana kadar bekle

# İsterseniz, yeni sesi bir dosyaya kaydedebilirsiniz
import soundfile as sf
sf.write("gurultulu_ses.wav", noisy_data, sr)
print("Gürültü eklenmiş ses kaydedildi: gurultulu_ses.wav")
