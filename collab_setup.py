import os
import subprocess
import zipfile

def install_requirements():
    """Install dependencies from requirements.txt"""
    req_file = os.path.join(project_dir, "requirements.txt")
    if os.path.exists(req_file):
        print(f"📦 Installing dependencies from {req_file} ...")
        subprocess.check_call(["pip", "install", "-r", req_file])
    else:
        print("⚠️ No requirements.txt found!")

def extract_zip_files():
    """Extract Posts.zip and Comments.zip inside data/"""
    data_dir = os.path.join(project_dir, "data")
    if not os.path.exists(data_dir):
        print("⚠️ No data directory found!")
        return

    for zip_name in ["Posts.zip", "Comments.zip"]:  # updated to match your repo
        zip_path = os.path.join(data_dir, zip_name)
        if os.path.exists(zip_path):
            print(f"📂 Extracting {zip_name} ...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(data_dir)
        else:
            print(f"⚠️ {zip_name} not found in {data_dir}")


if __name__ == "__main__":
    # Detect repo directory
    repo_name = "IR_Project01"
    cwd = os.getcwd()

    if os.path.basename(cwd) == repo_name:
        project_dir = cwd
    elif os.path.exists(os.path.join(cwd, repo_name)):
        project_dir = os.path.join(cwd, repo_name)
    else:
        raise FileNotFoundError(f"Could not find {repo_name} directory from {cwd}")

    print("📂 Project directory:", project_dir)

    # Run setup steps
    install_requirements()
    extract_zip_files()

    print("✅ Setup complete. You can now run your analysis scripts.")
