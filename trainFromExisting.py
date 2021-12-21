from fastai.vision.all import *

def label_func(x): return x.parent.name

def train():
    path = r'.\dataset'
    namesList = get_image_files(path)
    print(f"Total Images:{len(namesList)}")

    dls = ImageDataLoaders.from_path_func(path, namesList, label_func, bs=40)

    learn = cnn_learner(dls, resnet18, metrics=accuracy)
    print("Loaded")
    return(learn)

if __name__ == '__main__':
    learner = train()
    #interp = ClassificationInterpretation.from_learner(learner)
    #interp.plot_confusion_matrix(figsize=(6,6))
    #learner.lr_find()
    #learner.fine_tune(2, 3e-3)
    #learner.show_results()
    learner.fit(10)
    learner.export('jobData\model.pkl')