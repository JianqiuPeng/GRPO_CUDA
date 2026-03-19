import os
import shutil
import stat
import subprocess

def should_exclude(item):
    exclude = {
        ".git",
        ".gitmodules",
        "README.md",
        "LICENSE",
        "logs",
        "rl-trained-agents",
        "docs",
        "tests",
        "images",
        "docker",
        "scripts",
    }
    return item in exclude

def _handle_remove_readonly(func, path, exc_info):
    """
    Let rmtree remove read-only files/dirs on Windows.
    """
    if not os.path.exists(path):
        return
    os.chmod(path, stat.S_IWRITE)
    func(path)

def _remove_path(path):
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path, onerror=_handle_remove_readonly)
    elif os.path.exists(path):
        os.chmod(path, stat.S_IWRITE)
        os.remove(path)

def copy_backbone_files():
    backbone_dir = "./backbone"
    if not os.path.exists(backbone_dir):
        print("Backbone directory not found, skip copy (repository is already prepared).")
        return
    
    print("Copying files from backbone...")
    for item in os.listdir(backbone_dir):
        if should_exclude(item):
            print(f"Skipping excluded item: {item}")
            continue
            
        src = os.path.join(backbone_dir, item)
        dst = os.path.join(".", item)
        
        if os.path.isdir(src):
            if os.path.exists(dst):
                _remove_path(dst)
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns('.git', '.gitmodules'))
        else:
            if os.path.exists(dst):
                _remove_path(dst)
            shutil.copy2(src, dst)
    print("Files copied successfully.")

def apply_patch():
    patch_files = ["env_patch.patch", "param.patch"]
    for patch_file in patch_files:
        if not os.path.exists(patch_file):
            print(f"Patch file '{patch_file}' not found, skip.")
            continue
        
        print(f"Applying patch {patch_file}...")
        try:
            subprocess.run(["git", "apply", patch_file], check=True)
            print("Patch applied successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to apply patch: {e}")
        except FileNotFoundError:
            print("Git not found. Please install Git to apply patches.")

def main():
    try:
        copy_backbone_files()
        apply_patch()
        print("Setup finished.")
    except Exception as e:
        print(f"Setup failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
