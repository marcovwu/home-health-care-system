import os
import shutil


def check_dir(path, remove=False):
    if os.path.exists(path) and remove:
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

    return path