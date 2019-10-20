import os, shutil, random
from pathlib import Path

def split_data(sourcedir, traindir, testdir, split_size): #all arguments are in Pathlib format; not strings
    if not testdir.exists():
        testdir.mkdir(parents=True,exist_ok=True)

    if not traindir.exists():
        traindir.mkdir(parents=True,exist_ok=True)

    #all_list = [item for item in sourcedir.iterdir() if not item.stat().st_size == 0]
    all_list = [item for item in sourcedir.rglob("*.png")]
    print("all_list length", len(all_list))

    #Specifiy train size
    trainsize=int(split_size *len(all_list))
    print("trainsize", trainsize)

    #Specifiy test size
    testsize=int((1.0 - split_size) *len(all_list))
    print("testsize", testsize)

    #insert shuffled files
    testlist = random.sample(all_list,testsize)
    trainlist = random.sample(all_list,trainsize)

    #copy from the list to train/test directories
    for trainfile in trainlist:
        shutil.copy(str(trainfile), str(traindir)) #do not use shutil.move



    for testfile in testlist:
        shutil.copy(str(testfile), str(testdir)) #do not use shutil.move




imdir = Path.cwd()/"X_images"

for cls in imdir.iterdir():
    print("total images in "+cls.name+" ",len([item for item in cls.rglob("*")]))
    #break
testdir = imdir/"test"
traindir = imdir/"train"


##########################

#classes
#x1 =SigxFactor0.8
#x2 =SigxFactor1.0
#x3 =SigxFactor1.2

##########################

#classifiers source directory
x1_dir = imdir/"SigxFactor0.8"
x2_dir = imdir/"SigxFactor1.0"
x3_dir = imdir/"SigxFactor1.2"

#test directory
x1_testdir= Path.cwd()/"test_train_images"/"test"/"SigxFactor0.8"
x2_testdir= Path.cwd()/"test_train_images"/"test"/"SigxFactor1.0"
x3_testdir= Path.cwd()/"test_train_images"/"test"/"SigxFactor1.2"

#train directory
x1_traindir= Path.cwd()/"test_train_images"/"train"/"SigxFactor0.8"
x2_traindir= Path.cwd()/"test_train_images"/"train"/"SigxFactor1.0"
x3_traindir= Path.cwd()/"test_train_images"/"train"/"SigxFactor1.2"


split_size=0.9
split_data(x1_dir, x1_traindir, x1_testdir, split_size)
split_data(x2_dir, x2_traindir, x2_testdir, split_size)
split_data(x3_dir, x3_traindir, x3_testdir, split_size)


