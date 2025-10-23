import librosa
import numpy as np

def extract_melody_condition_musecontrol(audio_path, stereo=True):
    """
    提取 MuseControlLite 格式的 melody condition
    
    Parameters:
    -----------
    audio_path : str
        音频文件路径
    stereo : bool
        是否分别处理左右声道 (v3 版本，效果最好)
    
    Returns:
    --------
    melody_condition : np.ndarray
        如果 stereo=False: shape (T, 128)
        如果 stereo=True: shape (T, 8) - 左右声道各 top-4
    """
    # 载入音频 (44.1kHz, stereo)
    y, sr = librosa.load(audio_path, sr=44100, mono=False)
    
    # 确保是立体声
    if y.ndim == 1:
        y = np.stack([y, y])
    
    # CQT 参数（对应 MIDI 0-127）
    hop_length = 512
    n_bins = 128  # 对应 128 个 MIDI notes
    bins_per_octave = 12
    fmin = librosa.midi_to_hz(0)  # 8.176 Hz
    
    if stereo:
        # v3 版本：分别处理左右声道
        melody_conditions = []
        
        for channel in range(2):
            # 计算 CQT
            cqt = np.abs(librosa.cqt(
                y=y[channel],
                sr=sr,
                hop_length=hop_length,
                n_bins=n_bins,
                bins_per_octave=bins_per_octave,
                fmin=fmin
            ))  # shape: (128, T)
            
            # 提取每帧 top-4 最强的音高
            top4_indices = np.argsort(cqt, axis=0)[-4:, :]  # (4, T)
            
            melody_conditions.append(top4_indices.T)  # (T, 4)
        
        # 交错堆叠左右声道: [L1, L2, L3, L4, R1, R2, R3, R4]
        melody_condition = np.concatenate(melody_conditions, axis=1)  # (T, 8)
        
    else:
        # v2 版本：混合左右声道
        y_mono = librosa.to_mono(y)
        
        # 计算 CQT
        cqt = np.abs(librosa.cqt(
            y=y_mono,
            sr=sr,
            hop_length=hop_length,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            fmin=fmin
        ))  # shape: (128, T)
        
        # 提取每帧 top-4 最强的音高
        top4_indices = np.argsort(cqt, axis=0)[-4:, :]  # (4, T)
        melody_condition = top4_indices.T  # (T, 4)
    
    return melody_condition, cqt

def prepare_musecontrol_data(audio_path, output_path='melody_condition.npy'):
    """
    准备完整的 MuseControlLite 训练数据
    """
    import json
    
    # 1. 提取 melody condition
    melody_condition, cqt = extract_melody_condition_musecontrol(
        audio_path, 
        stereo=True
    )
    
    # 2. 保存为 numpy 格式
    np.save(output_path, melody_condition)
    
    # 3. 准备 JSON metadata (模仿 MuseControlLite 的格式)
    metadata = {
        "audio_path": audio_path,
        "melody_condition_path": output_path,
        "shape": melody_condition.shape,
        "format": "top4_128bin_CQT_stereo",
        "hop_length": 512,
        "sr": 44100,
        "n_bins": 128,
        "bins_per_octave": 12
    }
    
    with open(output_path.replace('.npy', '.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return melody_condition

# # 批量处理
target_dir = './home/fundwotsai/Deep_MIR_hw2/target_music_list_60s_index'
import os
audio_files = [
    os.path.join(target_dir, file) for file in os.listdir(target_dir) 
    if file.endswith('.mp3') or file.endswith('.wav')
]
os.makedirs('conditions', exist_ok=True)
for audio in audio_files:
    audio_name = os.path.basename(audio)
    print(f"Processing {audio_name}...")
    prepare_musecontrol_data(
        audio, 
        output_path=f'conditions/{audio_name}_melody_condition.npy'
    )

print("All melody conditions extracted and saved.")

# # 使用示例
# melody_condition, cqt_full = extract_melody_condition_musecontrol(
#     '/home/laiyian/DeepMIR-fall-2025_HW2/home/fundwotsai/Deep_MIR_hw2/target_music_list_60s_index/2.mp3', 
#     stereo=True  # 推荐用 v3 版本
# )

# print(f"Melody condition shape: {melody_condition.shape}")
# v3 输出: (T, 8) - 每个时间步有 8 个值（左右声道各 4 个）