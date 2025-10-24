import os
import json

caption_file = "caption_pair_clap_score.json"

output_file = "task2_suno_clap_score.json"

with open(caption_file, "r") as f:
    captions = json.load(f)

with open(output_file, "r") as f:
    datas = json.load(f)

new_datas = {}

for data in datas:
    # caption = captions[data["origin_target_song"]]["caption"]
    # origin_song_caption_clap_score = captions[data["origin_target_song"]]["clap_score"]
    new_datas[data] = {
        "origin_target_song": datas[data]["origin_target_song"],
        "matched_song_in_suno": datas[data]["matched_song_in_suno"],
        "matched_melodycondition": datas[data]["matched_melodycondition"],
        "matched_song_in_origin_dir": datas[data]["matched_song_in_origin_dir"],
        "caption": None,
        "origin_song_caption_clap_score": None,
        "melodycondition_song_caption_clap_score": None,
        "suno_song_caption_clap_score": None,
        "suno_clap_score_w_uncropped": datas[data]["suno_clap_score_w_uncropped"],
        "suno_clap_score_w_cropped": datas[data]["suno_clap_score_w_cropped"],
        "melodycondition_w_uncropped": datas[data]["melodycondition_clap_score"],
        "melodycondition_w_cropped": datas[data]["final_score"],
    }

with open("task2_suno_clap_score_with_caption.json", "w") as f:
    json.dump(new_datas, f, indent=4, ensure_ascii=False)
