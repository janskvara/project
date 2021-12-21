from fastai.vision.all import *
from pathlib import Path

def label_func(x): return x.parent.name

arch = resnet18

def train():
    path = r'.\dataset'
    namesList = get_image_files(path)
    print(f"Total Images:{len(namesList)}")

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
    learner.fit(1)
    modelpath = Path(__file__).parent.resolve() / f"jobData/model_{str(arch)}.pkl"
    print(modelpath)
    learner.export(modelpath)