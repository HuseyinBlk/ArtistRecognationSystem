import os
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sounddevice as sd
import librosa
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
import librosa.display

# Modeli yükleme
model = load_model("cnnmodelwithaugmentations1.h5")
singer_labels = {0: "Adele", 1: "Dua Lipa", 2: "Ed Sheeran", 3: "Hadise", 4: "Irem Derici",
                 5: "Murat Boz", 6: "Serdar Ortac", 7: "Tarkan", 8: "Taylor Swift", 9: "The Weeknd"}

# Spektrogram görüntüsünü oluşturma
def create_spectrogram(data, sr, save_path):
    """Ses verisinden mel-spektrogram oluştur ve kaydet."""
    spectrogram = librosa.feature.melspectrogram(y=data, sr=sr)
    plt.figure(figsize=(4, 3))  # Tkinter için uygun boyutta
    librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max),
                             y_axis='mel', fmax=8000, x_axis='time')
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Spektrogram görüntüsünü Tkinter arayüzüne ekleme
def show_spectrogram(image_path):
    """Spektrogramı GUI içinde gösterir."""
    clear_spectrogram()
    img = Image.open(image_path)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(img)
    ax.axis('off')
    canvas = FigureCanvasTkAgg(fig, master=spectrogram_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(pady=10)
    return canvas, canvas_widget

def load_image(image_path):
    """Mel-spektrogram görüntüsünü yükler ve normalize eder."""
    img = Image.open(image_path).convert('L')  # Gri tonlamaya çevir
    img = img.resize((128, 128))  # 128x128 boyutuna çevir
    return np.array(img) / 255.0

def predict_singer(audio_data, sr):
    """Ses verisinden tahmin yapar."""
    temp_spectrogram_path = "temp_spectrogram.png"
    create_spectrogram(audio_data, sr, temp_spectrogram_path)
    img = load_image(temp_spectrogram_path).reshape(1, 128, 128, 1)
    prediction = model.predict(img)
    predicted_label = np.argmax(prediction)
    return singer_labels[predicted_label], temp_spectrogram_path

def select_file():
    """Kullanıcının ses dosyası seçmesine izin verir."""
    file_path = filedialog.askopenfilename(title="Ses Dosyası Seç", filetypes=[("Ses Dosyaları", "*.wav *.mp3")])
    if file_path:
        try:
            audio_data, sr = librosa.load(file_path, duration=10)
            singer, spectrogram_path = predict_singer(audio_data, sr)
            result_label.config(text=f"Tahmin Edilen Şarkıcı: {singer}")
            show_spectrogram(spectrogram_path)
        except Exception as e:
            messagebox.showerror("Hata", f"Ses dosyası işlenirken bir hata oluştu: {e}")

def record_audio():
    """Mikrofonla ses kaydedip tahminde bulunur."""
    duration = 5  # 5 saniyelik kayıt
    sr = 22050
    messagebox.showinfo("Kayıt", "5 saniyelik kayıt başlayacak. Lütfen konuşun.")
    try:
        audio_data = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
        audio_data = audio_data.flatten()
        singer, spectrogram_path = predict_singer(audio_data, sr)
        result_label.config(text=f"Tahmin Edilen Şarkıcı: {singer}")
        show_spectrogram(spectrogram_path)
    except Exception as e:
        messagebox.showerror("Hata", f"Kayıt sırasında bir hata oluştu: {e}")

def clear_spectrogram():
    """Spektrogram penceresini temizler."""
    for widget in spectrogram_frame.winfo_children():
        widget.destroy()

# Tkinter Uygulama Penceresi
root = tk.Tk()
root.title("Şarkıcı Tanıma Uygulaması")
root.geometry("600x500")

# Üst bölüm: başlık ve düğmeler
title_label = tk.Label(root, text="Şarkıcı Tanıma Uygulaması", font=("Helvetica", 16))
title_label.pack(pady=10)

file_button = tk.Button(root, text="Ses Dosyası Seç", command=select_file, width=25, height=2)
file_button.pack(pady=10)

record_button = tk.Button(root, text="Mikrofon ile Ses Kaydet", command=record_audio, width=25, height=2)
record_button.pack(pady=10)

result_label = tk.Label(root, text="", font=("Helvetica", 12), fg="blue")
result_label.pack(pady=10)

# Alt bölüm: spektrogram görüntüsü
spectrogram_frame = tk.Frame(root)
spectrogram_frame.pack(pady=10)

# Ana döngü
root.mainloop()
