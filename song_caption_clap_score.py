import laion_clap
import numpy as np
import os
import json
from sklearn.metrics.pairwise import cosine_similarity

# calculate CLAP score between audio and its caption
model = laion_clap.CLAP_Module(enable_fusion=True)
model.load_ckpt()
target_dir = "DeepMIR-fall-2025_HW2/home/fundwotsai/Deep_MIR_hw2/target_music_list_60s_index"

with open("results.json", "r") as json_file:
    captions = json.load(json_file)

for key in captions.keys():
    print(f"Processing {key}...")
    caption = captions[key]
    caption_embed = model.get_text_embedding(x = [caption], use_tensor=False)[0]
    audio_path = os.path.join(target_dir, key)
    audio_embed = model.get_audio_embedding_from_filelist(x = [audio_path], use_tensor=False)[0]
    similarity = cosine_similarity(audio_embed.reshape(1, -1), caption_embed.reshape(1, -1))[0][0]
    captions[key] = {
        "caption": caption,
        "clap_score": round(float(similarity), 4)
    }

with open("caption_pair_clap_score.json", "w") as json_file:
    json.dump(captions, json_file, indent=4, ensure_ascii=False)