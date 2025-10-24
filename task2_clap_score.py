import laion_clap
import numpy as np
import librosa
import torch
import os
import json

uncropped_dir = "./home/fundwotsai/Deep_MIR_hw2/target_music_list_60s_index"
target_dir = "./home/fundwotsai/Deep_MIR_hw2/target_music_list_47s_index"

model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt()

target_files = [file for file in os.listdir(target_dir) if file.endswith(".wav") or file.endswith(".mp3")]

with open("task2_suno_clap_score_with_caption.json", "r") as json_file:
    datas = json.load(json_file)
    
# ref_files = [datas[file].get("matched_melodycondition", None) for file in target_files]
ref_files = [datas[file].get("matched_song_in_suno", None) for file in target_files]
uncropped_files = [os.path.join(uncropped_dir, datas[file].get("origin_target_song", None)) for file in target_files]
captions = [datas[file].get("caption", None) for file in target_files]

print(f"None captions: {sum(1 for c in captions if c is None)}")

target_files = [os.path.join(target_dir, file) for file in target_files]
target_audio_embed = model.get_audio_embedding_from_filelist(x = target_files, use_tensor=False)
uncropped_audio_embed = model.get_audio_embedding_from_filelist(x = uncropped_files, use_tensor=False)
ref_audio_embed = model.get_audio_embedding_from_filelist(x = ref_files, use_tensor=False)
captions_embed = model.get_text_embedding(captions, use_tensor=False)

print("Embeddings extracted.")

print("Calculating CLAP scores...")
from sklearn.metrics.pairwise import cosine_similarity

for i, target_file in enumerate(target_files):
    target_embed = target_audio_embed[i].reshape(1, -1)
    ref_embed = ref_audio_embed[i].reshape(1, -1)
    uncropped_embed = uncropped_audio_embed[i].reshape(1, -1)
    caption_embed = captions_embed[i].reshape(1, -1)
    cropped_similarity = cosine_similarity(target_embed, ref_embed)[0][0]
    uncropped_similarity = cosine_similarity(uncropped_embed, ref_embed)[0][0]
    caption_similarity = cosine_similarity(target_embed, caption_embed)[0][0]
    # only store with resolution 4 decimal places
    # datas[os.path.basename(target_file)]["melodycondition_w_uncropped"] = round(float(cropped_similarity), 4)
    # datas[os.path.basename(target_file)]["melodycondition_w_cropped"] = round(float(uncropped_similarity), 4)
    # datas[os.path.basename(target_file)]["melodycondition_song_caption_clap_score"] = round(float(caption_similarity), 4)
    datas[os.path.basename(target_file)]["suno_clap_score_w_cropped"] = float(f"{cropped_similarity:.4f}")
    datas[os.path.basename(target_file)]["suno_clap_score_w_uncropped"] = float(f"{uncropped_similarity:.4f}")
    datas[os.path.basename(target_file)]["suno_song_caption_clap_score"] = round(float(caption_similarity), 4)

with open("task2_suno_clap_score_with_caption.json", "w") as json_file:
    json.dump(datas, json_file, indent=4, ensure_ascii=False)

print("CLAP scores computed and saved to suno_clap_score.json")

