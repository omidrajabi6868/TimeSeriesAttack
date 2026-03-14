import numpy as np
from Dataset.DataManagement import ImageDataSet
from ImageClassification import ClassificationBase


def main():
    label_path = "/home/oraja001/Jlab/Hydra data/labels_v2.txt"

    dataset = ImageDataSet(label_path=label_path, trasform=None)

    classification = ClassificationBase('ResNet50', optimizer_name='Adam')

    
    classification.train_model()
    

    
    return

if __name__== "__main__":
    main()