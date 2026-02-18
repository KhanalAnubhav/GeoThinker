import json
import os

def sanitize(input_path, output_path):
    print(f"Sanitizing {input_path} -> {output_path}")
    if not os.path.exists(input_path):
        print(f"Error: {input_path} does not exist.")
        return
    needed_keys = {'id', 'question', 'images', 'gt_answer'}
    count = 0
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            try:
                data = json.loads(line)
                # Only keep what we need
                clean_data = {k: data[k] for k in needed_keys if k in data}
                f_out.write(json.dumps(clean_data) + '\n')
                count += 1
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line in {input_path}")
    print(f"Successfully processed {count} lines.")

base_path = "/mnt/4TB_HDD/scene_understanding/MindCube/raw/"
if __name__ == "__main__":
    sanitize(os.path.join(base_path, "MindCube_train.jsonl"), os.path.join(base_path, "MindCube_train_clean.jsonl"))
    sanitize(os.path.join(base_path, "MindCube_tinybench.jsonl"), os.path.join(base_path, "MindCube_tinybench_clean.jsonl"))
