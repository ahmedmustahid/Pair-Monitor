import os, shutil

def makeDirectory(dirname):
     if not os.path.exists(dirname):
         os.makedirs(dirname)

def compress_files():
    basedir="."

    bases=os.listdir(basedir)
    #print bases
    for base in bases:
        if os.path.isdir(base) and base.split('_')[0]=="Run":
            image ="./"+ base+"/"+"images/"
            print "image "+image
            if os.path.exists(image):
                basename= "compressed_images"+"/"+base
                root_dir= basedir
                base_dir = image
                #imdir= basedir+"/"+"compressed_images"+"/"+base+"/"+base
                makeDirectory("compressed_images/")
                #print "imdir "+imdir
                #images= os.listdir(image)
                shutil.make_archive(basename,'gztar',root_dir,base_dir)
                #break

                #for imagefile in images:
                #    imagefiledir=image+"/"+imagefile
                #    imagefiledir = os.path.abspath(imagefiledir)



compress_files()
