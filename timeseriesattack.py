import torch
import numpy as np

from Dataset.DataManagement import TimeSeriesDataset
from Attacks.TimeSeriesAdversarialAttack import AdversarialAttack
from Attacks.TimeSeriesBackdoorAttack import BackdoorAttack
from Tasks.TimeSeriesForecasting import ForecastBase

def main():

    task = 'adversarial_attack'
    input_len = 96
    output_len = 96

    train_original_model = False
    
    dataset = TimeSeriesDataset(
        csv_path="/home/oraja001/Jlab/Sensor data/Original Data/2022_OUTPUT_VARS.1h.csv",
        timestamp_col='DATE_TIME',
        input_len=input_len,
        output_len=output_len,
        input_cols=['IBC1H04CRCUR2', 'MMSHLAE', 'BLA', 'rad48_p1', 'rad44_p1', 'rad29_p1'],
        output_cols=['rad48_p1', 'rad44_p1', 'rad29_p1'],
        freq="H",
        stride=8,
        train_ratio=0.8,
        val_ratio=0.15,
        add_time_features=True,
        normalize=False,
        zero_threshold=1e-4,
        var_threshold=1e-4
    )

    train_loader, val_loader, test_loader = dataset.get_dataloaders(batch_size=32)

    sample_inputs, _ = next(iter(train_loader))
    num_vars = sample_inputs.shape[-1]

    forecast = ForecastBase(model_name='PatchTST', 
                            optimizer_name='Adam', 
                            checkpoint_dir='backups/forecast',
                            input_len=input_len,
                            output_len=output_len,
                            num_vars=num_vars)
    
    if train_original_model:
            forecast.train_model(
            train_loader,
            val_loader,
            learning_rate=1e-4,
            epoch_num=300,
            resume=False,
            resume_from='backups/last_checkpoint.pth',
        )
    else:
        forecast.load_checkpoint("backups/forecast/best_checkpoint.pth")
    
    test_metrics = forecast.evaluate_model(test_loader=test_loader)
    print(f'test_loss: {test_metrics["loss"]}')


    return



if __name__ == '__main__':
    main()