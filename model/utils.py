import os

def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def path_and_name(path: str):
    dir_path = os.path.dirname(path)
    name, _ = os.path.splitext(os.path.basename(path))
    return dir_path, name