# collab_setup.py
import os
import zipfile
import subprocess
import sys

# =========================
# 1️⃣ INSTALL DEPENDENCIES
# =========================
def install_dependencies(requirements_file="requirements.txt"):
    """
    Installs all packages listed in requirements.txt using pip.
    """
    if not os.path.exists(requirements_file):
        print(f"Requirements file '{requirements_file}' not found!")
        return
    print("Installing dependencies from", requirements_file)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])


# =========================
# 2️⃣ EXTRACT ZIP FILES
# =========================
def extract_zip(zip_path, extract_to=None):
    """
    Extracts a zip file to the specified directory.
    """
    if not os.path.exists(zip_path):
        print(f"ZIP file '{zip_path}' not found! Skipping extraction.")
        return
    if extract_to is None:
        extract_to = os.path.dirname(zip_path)
    
    print(f"Extracting '{zip_path}' to '{extract_to}' ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extraction complete: '{zip_path}'")


# =========================
# 3️⃣ MAIN SETUP FUNCTION
# =========================
def main():
    # Install dependencies
    install_dependencies("requirements.txt")

    # Create data folder if it doesn't exist
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Extract posts.zip and comments.zip
    posts_zip = os.path.join(data_dir, "posts.zip")
    comments_zip = os.path.join(data_dir, "comments.zip")

    extract_zip(posts_zip, data_dir)
    extract_zip(comments_zip, data_dir)

    print("\n✅ Colab setup complete! You can now run your scripts.")


if __name__ == "__main__":
    main()
