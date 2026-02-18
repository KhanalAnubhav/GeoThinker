from huggingface_hub import snapshot_download
import os
import zipfile

# Define the specific path where you want the data stored
target_path = "/mnt/4TB_HDD/scene_understanding/"  # Change this to your desired path

# Create the directory if it doesn't exist (optional but good practice)
os.makedirs(target_path, exist_ok=True)

# Login is still required if the dataset is gated
# from huggingface_hub import login
# login()

print(f"Downloading MindCube raw data to {target_path}...")

# Download the repository directly to get raw files (data.zip, etc.)
snapshot_download(
    repo_id="MLL-Lab/MindCube",
    repo_type="dataset",
    local_dir=target_path,
    local_dir_use_symlinks=False
)

# Extract data.zip if it exists
zip_path = os.path.join(target_path, "data.zip")
if os.path.exists(zip_path):
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_path)
    print("Extraction complete.")
else:
    print("data.zip not found in the downloaded files.")

print(f"Dataset files are available at {target_path}")