import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def preprocess_data(train_data, test_data):

    train_data = process(train_data)
    test_data = process(test_data)

    return train_data, test_data


def process(train_data):

    train_data.dropna(inplace=True)

    train_data['DateTime'] = pd.to_datetime(train_data['DateTime'], format='%Y-%m-%d %H:%M:%S')

    for col in train_data.columns:
        if col != 'DateTime':
            train_data[col] = pd.to_numeric(train_data[col], errors='coerce')
    train_data['Date'] = train_data['DateTime'].dt.date

    train_data['Sub_metering_remainder'] = (train_data['Global_active_power'] * 1000 / 60) \
                                        - (train_data['Sub_metering_1'] 
                                        + train_data['Sub_metering_2'] 
                                        + train_data['Sub_metering_3'])

    train_data['Voltage']=train_data['Voltage'].astype(str).str.extract(r'^(\d+\.\d{2})').astype(float)

    daily_train = train_data.groupby('Date', as_index=False).agg({
        'Global_active_power': 'sum',
        'Global_reactive_power': 'sum',
        'Sub_metering_1': 'sum',
        'Sub_metering_2': 'sum',
        'Sub_metering_3': 'sum',
        'Sub_metering_remainder': 'sum',
        'Voltage': 'mean',
        'Global_intensity': 'mean',
        'RR': 'first',
        'NBJRR1': 'first',
        'NBJRR5': 'first',
        'NBJRR10': 'first',
        'NBJBROU': 'first'
    })

    daily_train.sort_values(by='Date', inplace=True)

    return daily_train

if __name__ == '__main__':
    train_data = pd.read_csv('train.csv', dtype=str)
    test_data = pd.read_csv('test.csv', dtype=str)
    train_data, test_data = preprocess_data(train_data, test_data)
    print(train_data.head())
    print(test_data)