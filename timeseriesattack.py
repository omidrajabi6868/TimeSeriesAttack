import torch
import numpy as np

from Dataset.DataManagement import TimeSeriesDataset
from Attacks.TimeSeriesAdversarialAttack import AdversarialAttack
from Attacks.TimeSeriesBackdoorAttack import BackdoorAttack
from Tasks.TimeSeriesForecasting import ForecastBase

def main():

    task = 'adversarial_attack'
    input_len = 24
    output_len = 12
    
    dataset = TimeSeriesDataset(
        cvs_path="/home/oraja001/Jlab/Sensor data/Original Data/2022_OUTPUT_VARS.1h.csv",
        timestamp_col='DATE_TIME',
        input_len=input_len,
        output_len=output_len,
        input_cols=['IBC1H04CRCUR2', 'MMSHLAE', 'BLA', 'rad48_p1', 'rad44_p1', 'rad29_p1'],
        output_cols=['rad48_p1', 'rad44_p1', 'rad29_p1'],
        freq="H",
        stride=1,
        train_ratio=0.7,
        val_ratio=0.15,
        add_time_features=True,
        normalize=True,
        zero_threshold=1e-4,
        var_threshold=1e-5
    )

    train_loader, val_loader, test_loader = dataset.get_dataloaders(batch_size=32)




    
    
    
    
    
    return






if __name__ == '__main__':
    main()