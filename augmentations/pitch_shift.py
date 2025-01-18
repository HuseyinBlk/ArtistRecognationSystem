import librosa
import soundfile as sf
import os

def change_gender_batch(input_folder, n_steps_female=3, n_steps_male=-3):
    """
    Bir klasördeki tüm WAV dosyalarının tonunu değiştirerek cinsiyet etkisi yaratır.

    Parametreler:
    input_folder (str): Giriş ses dosyalarının bulunduğu klasör.
    n_steps_female (int): Kadın sesine dönüştürmek için yarım ton artışı.
    n_steps_male (int): Erkek sesine dönüştürmek için yarım ton düşüşü.
    """
    try:
        # Klasördeki tüm dosyaları listele
        for file_name in os.listdir(input_folder):
            if file_name.endswith(".wav"):
                file_path = os.path.join(input_folder, file_name)

                # Ses dosyasını yükleme
                audio, sr = librosa.load(file_path, sr=None)

                # Kadın sesine dönüştürme
                transformed_audio_female = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps_female)
                female_output_file = os.path.join(input_folder, file_name.replace(".wav", "_female.wav"))
                sf.write(female_output_file, transformed_audio_female, sr)

                # Erkek sesine dönüştürme
                transformed_audio_male = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps_male)
                male_output_file = os.path.join(input_folder, file_name.replace(".wav", "_male.wav"))
                sf.write(male_output_file, transformed_audio_male, sr)

                print(f"Dönüştürüldü: {file_name} -> {female_output_file}, {male_output_file}")
    except Exception as e:
        print(f"Hata: {e}")

# Kullanım örneği
input_folder = r"C:\\Users\\Administrator\\Desktop\\dataset1\\The_Weeknd"  # Giriş klasörü
change_gender_batch(input_folder)
