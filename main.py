import numpy as np
from Dataset.DataManagement import ImageDataSet
from ImageClassification import ClassificationBase


def main():
    label_path = "/home/oraja001/Jlab/Hydra data/labels_v2.txt"
    image_size = (640, 288)
    
    dataset = ImageDataSet(label_path=label_path, transform=None, image_size=image_size)
    train_loader, val_loader, test_loader = dataset.train_val_test_loader(batch_size=64)

    classification = ClassificationBase(
        model_name='ResNet18',
        optimizer_name='Adam',
        checkpoint_dir='backups',
    )

    # classification.train_model(train_loader, val_loader, learning_rate=1e-4, epoch_num=10)

    # Resume example:
    # classification.train_model(
    #     train_loader,
    #     val_loader,
    #     learning_rate=1e-4,
    #     epoch_num=20,
    #     resume_from='backups/last_checkpoint.pth',
    # )

    classification.load_checkpoint("backups/best_checkpoint.pth")

    test_metrics = classification.evaluate_model(test_loader=test_loader)
    print(f'test_loss: {test_metrics["loss"]}, test_accuracy: {test_metrics["accuracy"]}')
    
    return

if __name__ == "__main__":
    main()