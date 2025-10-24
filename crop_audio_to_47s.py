import soundfile as sf
import os

target_dir = './home/fundwotsai/Deep_MIR_hw2/target_music_list_60s_index'
output_dir = './home/fundwotsai/Deep_MIR_hw2/target_music_list_47s_index'
os.makedirs(output_dir, exist_ok=True)
audio_files = [
    os.path.join(target_dir, file) for file in os.listdir(target_dir) 
    if file.endswith('.mp3') or file.endswith('.wav')
]

for audio_path in audio_files:
    audio, sr = sf.read(audio_path)
    if audio.shape[0] > sr * 47:
        cropped_audio = audio[:sr * 47]
    else:
        cropped_audio = audio
    if audio_path.endswith('.mp3'):
        output_path = os.path.join(output_dir, os.path.basename(audio_path).replace('.mp3', '.wav'))
    else:
        output_path = os.path.join(output_dir, os.path.basename(audio_path))
    sf.write(output_path, cropped_audio, sr)
    print(f"Cropped and saved: {output_path}")