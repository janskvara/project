from fastai.vision.all import *
from pathlib import Path

def label_func(x): return x.parent.name

arch = xse_resnext18_deeper
epochs = 5
archname = str(arch)
name = archname[10:int(len(archname))-23]

def train():
    path = r'.\dataset84'
    namesList = get_image_files(path)
    print(f"Total Images:{len(namesList)}")

    dls = ImageDataLoaders.from_path_func(path, namesList, label_func, bs=32)

    learn = cnn_learner(dls, arch, pretrained=False, metrics=accuracy)
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