import json
import os

with open("task2_suno_clap_score.json", "r") as json_file:
    datas = json.load(json_file)

new_datas = {}

for i, data in enumerate(datas):
    new_datas[data] = {
        "target_song": data,
        "matched_song_in_suno": datas[data]["matched_song_in_suno"],
        "matched_melodycondition": f"melodycondition/test_{i}.wav",
        "matched_song_in_origin_dir": datas[data]["matched_song_in_origin_dir"],
        "suno_clap_score": datas[data]["suno_clap_score"],
        "melodycondition_clap_score": None
    }

with open("task2_suno_clap_score.json", "w") as json_file:
    json.dump(new_datas, json_file, indent=4, ensure_ascii=False)