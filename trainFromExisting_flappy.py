from fastai.vision.all import *
from pathlib import Path

def label_func(x): return x.parent.name

arch = resnet34
epochs = 5
archname = str(arch)
name = archname[10:int(len(archname))-23]

def train():
    path = r'.\data_flappy'
    namesList = get_image_files(path)
    print(f"Total Images:{len(namesList)}")

    # # checks if each image has resolution of 800x200, if not, resizes it
    # for item in namesList:
    #     img = Image.open(item)
    #     wid, hgt = img.size
    #     if wid != 800 or hgt != 200:
    #         img_res = img.resize((800, 200))
    #         img_res.save(item)

    dls = ImageDataLoaders.from_path_func(path, namesList, label_func, bs=16, num_workers=0) # batchsize setting

    learn = cnn_learner(dls, arch, pretrained=True, metrics=accuracy)
    print("Loaded")
    return(learn)

if __name__ == '__main__':

    learner = train()
    learner.fit(epochs)

    modelpath = Path(__file__).parent.resolve() / f"jobData_flappy/model_{name}-{str(epochs)}.pkl"
    learner.export(modelpath)

    learner.show_results()
    interp = ClassificationInterpretation.from_learner(learner)
    interp.plot_confusion_matrix(figsize=(6,6))
    interp.plot_top_losses(20, figsize=(10,10))
    try:
        interp.print_classification_report()
    except:
        pass

    #learner.lr_find()
    #learner.fine_tune(2, 3e-3)
    