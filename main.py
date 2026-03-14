import numpy as np
from Dataset.DataManagement import ImageDataSet
from ImageClassification import ClassificationBase


def main():
    label_path = "/home/oraja001/Jlab/Hydra data/labels_v2.txt"
    image_size = (224, 224)
    
    dataset = ImageDataSet(label_path=label_path, transform=None, image_size=image_size)

    cl = ClassificationBase('ResNet50', optimizer_name='Adam')

    train_loader, val_loader, test_loader = dataset.train_val_test_loader()

    cl.train_model(
        train_loader=test_loader,
        val_loader=val_loader,
        learning_rate=1e-4,
        epoch_num=10
    )

    cl.evaluate_model(cl.model, test_loader=test_loader)
    
    return

if __name__== "__main__":
    main()