from fastai.vision.all import *
import pickle as pk

def label_func(x): return x.parent.name

def export(data):
    pk_out = open('learner.pkl', 'wb')
    pk.dump(data, pk_out)
    pk_out.close()

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

    learner.fit(3)
    export(learner)