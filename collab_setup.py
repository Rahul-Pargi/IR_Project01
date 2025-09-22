import os
import subprocess
import zipfile

# Get repo root (folder where this script resides)
repo_root = os.path.dirname(os.path.abspath(__file__))

# Path to requirements.txt
requirements_path = os.path.join(repo_root, "requirements.txt")

# Install dependencies
print("\nInstalling dependencies from requirements.txt")
subprocess.run(["pip", "install", "-r", requirements_path], check=True)

# Extract ZIP files from data/
data_folder = os.path.join(repo_root, "data")
zip_files = ["posts.zip", "comments.zip"]

for zf in zip_files:
    zip_path = os.path.join(data_folder, zf)
    if os.path.exists(zip_path):
        print(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_folder)
    else:
        print(f"ZIP file '{zip_path}' not found! Skipping extraction.")

print("\n✅ Colab setup complete! You can now run your scripts.")
