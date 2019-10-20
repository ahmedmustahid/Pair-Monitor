from pathlib import Path
from shutil import unpack_archive


path = Path.cwd()

for child in path.iterdir():
    #print(child.suffix)
    if(child.name=="compressed_images"):
        print(child.name)
        newp= path/child.name
        #print(newp)
        for childer in newp.iterdir():
            print(childer)
            extract_dir="./images"
            unpack_archive(childer,extract_dir)
            print("uncompressed to ",str(extract_dir))
            #dirs = [e for e in childer.iterdir() if e.is_file()]
            #print(dirs)
            #for e in childer.rglob("*.tar.gz"):
            #    print(e)

