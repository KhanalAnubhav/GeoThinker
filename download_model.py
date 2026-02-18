from huggingface_hub import snapshot_download
import os
import shutil

# Define the repository and the specific subfolder for the 8B model
repo_id = "lihy285/GeoThinker"
subfolder = "GeoThinker-VGGT-Scaled-Qwen3VL-8B"
local_dir = "GeoThinker-8B"

print(f"Downloading model {subfolder} from {repo_id}...")

# Download the specific subfolder
# We use allow_patterns to only get the files in that subfolder
download_path = snapshot_download(
    repo_id=repo_id,
    allow_patterns=f"{subfolder}/*",
    local_dir="hf_download_tmp"
)

# Move the contents of the subfolder to the target local_dir
src_dir = os.path.join("hf_download_tmp", subfolder)
if os.path.exists(local_dir):
    print(f"Directory {local_dir} already exists. Removing it to ensure a clean download.")
    shutil.rmtree(local_dir)

print(f"Moving weights from {src_dir} to {local_dir}...")
shutil.move(src_dir, local_dir)

# Clean up
shutil.rmtree("hf_download_tmp")

print(f"Model successfully downloaded to {os.path.abspath(local_dir)}")
