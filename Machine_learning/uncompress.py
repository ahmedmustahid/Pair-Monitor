import os, shutil
from pathlib import Path

cmprs_path=Path.cwd()/'compressed_images'
uncmprs_path = Path.cwd()/'images'
uncmprs_path.mkdir(parents=True,exist_ok=True)

for f in cmprs_path.iterdir():
    print(f)
    shutil.unpack_archive(f,uncmprs_path,'gztar')

