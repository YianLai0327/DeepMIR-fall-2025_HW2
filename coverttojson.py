import os
import json

with open("best_matches.txt", "r") as f:
    best_matches = f.readlines()

dictionary = {}

best_matches = best_matches[1:]  # skip header line

for line in best_matches:
    splits = line.strip().split(',')
    target_file = splits[:-2]
    target_file = ','.join(target_file)
    reference_file = splits[-2]
    similarity_score = splits[-1]
    dictionary[target_file] = {
        "reference_file": reference_file,
        "similarity_score": float(similarity_score)
    }

with open("best_matches.json", "w") as json_file:
    json.dump(dictionary, json_file, indent=4, ensure_ascii=False)