import torch
import numpy as np

from Dataset.DataManagement import TimeSeriesDataset
from Attacks.TimeSeriesAdversarialAttack import AdversarialAttack
from Attacks.TimeSeriesBackdoorAttack import BackdoorAttack

def main():

    task = 'adversarial_attack'

    data_path = "/home/oraja001/Jlab/Sensor data/Original Data/2022_OUTPUT_VARS.1h.csv"
    timestamp_col = 'DATE_TIME'
    input_len = 24
    output_len = 12
    input_cols = ['IBC1H04CRCUR2', 'MMSHLAE', 'BLA', 'rad48_p1', 'rad44_p1', 'rad29_p1']
    output_cols = ['rad48_p1', 'rad44_p1', 'rad29_p1']
    freq="H"
    stride=1
    add_time_features=True
    normalize=True
    
    dataset = TimeSeriesDataSet(cvs_path=data_path,
                                timestamp_col=timestamp_col,
                                input_len=input_len,
                                output_len=output_len,
                                input_cols=input_cols,
                                output_cols=output_cols,
                                freq=freq,
                                stride=stride,
                                add_time_features=add_time_features,
                                normalize=normalize)
    
    
    
    
    return






if __name__ == '__main__':
    main()