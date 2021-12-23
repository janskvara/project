from fastai.vision.all import *
from pathlib import Path
from PIL import Image

def label_func(x): return x.parent.name

arch = resnet18
epochs = 5
archname = str(arch)
name = archname[10:int(len(archname))-23]

def train():
    path = r'.\dataset'
    namesList = get_image_files(path)
    print(f"Total Images:{len(namesList)}")

    # checks if each image has resolution of 800x200, if not, resizes it
    for item in namesList:
        img = Image.open(item)
        wid, hgt = img.size
        if wid != 800 or hgt != 200:
            img_res = img.resize((800, 200))
            img_res.save(item)

    dls = ImageDataLoaders.from_path_func(path, namesList, label_func, bs=16)

    learn = cnn_learner(dls, arch, pretrained=True, metrics=accuracy)
    print("Loaded")
    return(learn)

if __name__ == '__main__':
    learner = train()
    #interp = ClassificationInterpretation.from_learner(learner)
    #interp.plot_confusion_matrix(figsize=(6,6))
    #learner.lr_find()
    #learner.fine_tune(2, 3e-3)
    #learner.show_results()
    learner.fit(epochs)
    modelpath = Path(__file__).parent.resolve() / f"jobData/model_{name}-{str(epochs)}.pkl"
    print(modelpath)
    learner.export(modelpath)