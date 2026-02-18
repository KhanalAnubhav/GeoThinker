import os
import json
import subprocess
import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image

BASE_PATH = "/mnt/4TB_HDD/scene_understanding/dataset/Interior_GS"
OUTPUT_DIR = BASE_PATH # As requested by user

TASK_CONFIGS = {
    "existence": "existence/existence_qa.json",
    "absolute_distance": "vsi/absolute_distance_qa.json",
    "object_size": "vsi/object_size_qa.json",
    "object_count": "vsi/object_count_qa.json",
    "relative_direction": "vsi/relative_direction_qa.json", 
    "relative_distance": "vsi/relative_distance_qa.json",
    "reasoning": "reasoning/reasoning_qra_triplets.json"
}

def is_solid_color(image_path):
    try:
        img = Image.open(image_path).convert("L")
        data = np.array(img)
        # Check standard deviation of colors
        std_dev = np.std(data)
        if std_dev < 1.0: # Very close to solid color
            return True
        return False
    except Exception as e:
        print(f"Error checking solid color for {image_path}: {e}")
        return True

def get_sharpness_score(image_path):
    try:
        # Load as grayscale
        img = Image.open(image_path).convert("L")
        data = np.array(img, dtype=np.float64)
        
        # Laplacian kernel approximation using numpy roll for fast convolution
        laplacian = (
            np.roll(data,  1, axis=0) + 
            np.roll(data, -1, axis=0) + 
            np.roll(data,  1, axis=1) + 
            np.roll(data, -1, axis=1) - 
            4 * data
        )
        return np.var(laplacian)
    except Exception as e:
        print(f"Error calculating sharpness for {image_path}: {e}")
        return 0

def extract_frames(video_path, output_folder, sample_count=100, target_count=16):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # We use a temporary subdirectory for the 100 samples
    temp_folder = os.path.join(output_folder, "temp_samples")
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    # Get video duration using ffprobe
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path
    ]
    try:
        duration_out = subprocess.check_output(cmd).decode().strip()
        duration = float(duration_out)
    except Exception as e:
        print(f"Error getting duration for {video_path}: {e}")
        return []

    # 1. Extract 100 candidate frames
    timestamps = [i * (duration / sample_count) for i in range(sample_count)]
    
    candidates = []
    
    print(f"Sampling {sample_count} frames from {os.path.basename(video_path)}...")
    for i, ts in enumerate(timestamps):
        frame_name = f"sample_{i:04d}.jpg"
        frame_path = os.path.join(temp_folder, frame_name)
        
        if not os.path.exists(frame_path):
            extract_cmd = [
                "ffmpeg", "-ss", str(ts), "-i", video_path,
                "-frames:v", "1", "-q:v", "2", frame_path, "-y", "-loglevel", "error"
            ]
            subprocess.run(extract_cmd)
        
        # 2. Score and Filter
        if os.path.exists(frame_path):
            if not is_solid_color(frame_path):
                score = get_sharpness_score(frame_path)
                candidates.append((frame_path, score, i))
            else:
                os.remove(frame_path) # Clean up solid color frames immediately
    
    # 3. Select top N sharpest frames
    # Sort by score descending
    candidates.sort(key=lambda x: x[1], reverse=True)
    best_candidates = candidates[:target_count]
    
    # Sort by index to preserve temporal order
    best_candidates.sort(key=lambda x: x[2])
    
    final_frame_files = []
    for i, (old_path, score, _) in enumerate(best_candidates):
        new_name = f"frame_{i:04d}.jpg"
        new_path = os.path.join(output_folder, new_name)
        
        # Copy the best frame to the final location
        shutil.copy(old_path, new_path)
        final_frame_files.append(os.path.join("frames", new_name))
    
    # 4. Cleanup temp folder
    shutil.rmtree(temp_folder)
    
    return final_frame_files

def main():
    # Only pick directories that look like scene IDs (numeric or alphanumeric)
    scenes = [d for d in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, d)) and "_" in d]
    
    aggregated_tasks = {task: [] for task in TASK_CONFIGS}
    
    for scene in tqdm(scenes, desc="Processing scenes"):
        scene_path = os.path.join(BASE_PATH, scene)
        
        # Find video
        video_file = None
        for ext in ["mp4", "mov", "webm"]:
            potential_file = os.path.join(scene_path, f"scene.{ext}")
            if os.path.exists(potential_file):
                video_file = potential_file
                break
        
        if not video_file:
            continue
            
        # Extract frames
        frame_folder = os.path.join(scene_path, "frames")
        extract_frames(video_file, frame_folder, 100)
        
        for task_name, rel_path in TASK_CONFIGS.items():
            task_json_path = os.path.join(scene_path, rel_path)
            if os.path.exists(task_json_path):
                try:
                    with open(task_json_path, "r") as f:
                        data = json.load(f)
                        if not isinstance(data, list):
                            data = [data]
                            
                        for i, item in enumerate(data):
                            # Construct standardized item
                            processed_item = {
                                "id": f"{scene}_{task_name}_{i}",
                                "question": item.get("question"),
                                "images": scene, # Relative path to the scene folder from BASE_PATH
                                "gt_answer": item.get("answer"),
                                "question_type": task_name
                            }
                            # Preserve extra fields
                            for k, v in item.items():
                                if k not in ["question", "answer"]:
                                    processed_item[k] = v
                                    
                            aggregated_tasks[task_name].append(processed_item)
                except Exception as e:
                    print(f"Error reading {task_json_path}: {e}")

    # Write aggregated files
    for task_name, items in aggregated_tasks.items():
        if not items:
            continue
        output_file = os.path.join(OUTPUT_DIR, f"interior_gs_{task_name}.jsonl")
        with open(output_file, "w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")
        print(f"Wrote {len(items)} items to {output_file}")

if __name__ == "__main__":
    main()
