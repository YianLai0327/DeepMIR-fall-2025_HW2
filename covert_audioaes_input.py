import os
import json

input_file = "task1_output.jsonl"

output_file = "task1_output_4.jsonl"

with open(input_file, "r", encoding="utf-8") as f_in, \
     open(output_file, "w", encoding="utf-8") as f_out:
    for line in f_in:
        data = json.loads(line)
        # 將每個數值四捨五入到小數點後第四位
        rounded_data = {key: round(value, 4) for key, value in data.items()}
        f_out.write(json.dumps(rounded_data, ensure_ascii=False) + "\n")


print("四捨五入完成！")