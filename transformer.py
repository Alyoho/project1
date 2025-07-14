import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math
import time
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline 

from preprocess import preprocess_data
from postprocess import postprocess


class PE(nn.Module):
    
    def __init__(self, d_model, max_len=5000):
        
        super(PE, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
    
    
class TransformerTimeSeriesModel(nn.Module):
    def __init__(self, input_dim, output_dim, seq_length, output_length, 
                 d_model = 64, nhead = 8, num_layers = 4, dropout=0):
        super(TransformerTimeSeriesModel, self).__init__()
        self.output_length = output_length
        self.output_dim = output_dim
        self.src_mask = None
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_coding = PE(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model = d_model, nhead = nhead, 
                                                        dim_feedforward=4 * d_model, dropout = dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers = num_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(
            self.decoder_layer, 
            num_layers=num_layers
        )
        self.tgt_pos_encoding = PE(d_model)
        self.tgt_embedding = nn.Linear(output_dim, d_model)
        
        self.fc_out = nn.Linear(d_model, output_dim)
        self.encoder_mask = self.generate_mask(seq_length)
        self.decoder_mask = self.generate_mask(output_length)

        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=2,  # 保持序列长度不变
            dilation=2   # 扩大感受野
        )
    

    def generate_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
    
    def forward(self, src):

        src = self.embedding(src)
        src = self.pos_coding(src)
        memory = self.transformer_encoder(src, self.encoder_mask.to(src.device))
        
        # 生成初始目标序列（零张量）
        tgt = torch.zeros(self.output_length, src.size(1), self.output_dim, device=src.device)
        tgt = self.tgt_embedding(tgt)
        tgt = self.tgt_pos_encoding(tgt)
        
        # 解码器处理
        output = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=self.decoder_mask.to(src.device)
        )
        
        # 最终输出处理
        output = self.fc_out(output)
        return output.permute(1, 0, 2)  # [batch, output_length, output_dim]
    
    def init_weights(self):
        initrange = 0.1
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.uniform_(-initrange, initrange)
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    

class get_dataset(Dataset):
    
    def __init__(self, data, seq_length, label_length, features):
        self.data = data
        self.features = features
        self.seq_length = seq_length
        self.label_length = label_length
        self.data, self.data_mean, self.data_std = self.get_data()
 
    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, index):
        en_input = self.data[index, :self.seq_length, :]
        label = self.data[index, -self.label_length:, -1].unsqueeze(1)
        return en_input, label
               
    def get_data(self):
        data = self.data
        data.index = data['Date']
        data = data.drop('Date', axis=1)
        data_mean = data.mean()
        data_std = data.std()
        for col in data.columns:
            if data_std[col] == 0:
                print(f"Warning: Column '{col}' has zero variance and will not be normalized.")
            else:
                data[col] = (data[col] - data_mean[col]) / data_std[col]
        num_sample = len(data) - self.seq_length - self.label_length + 1
        print('len(data):', len(data), 'num_sample:', num_sample)
        print('len(self.features):',len(self.features))
        seq_data = torch.zeros(num_sample, self.seq_length + self.label_length, len(self.features))
 
        for i in range(num_sample):
            col_data = data.loc[:, self.features]

            seq_data[i] = torch.tensor(col_data.iloc[i:i + self.seq_length + self.label_length].values)
 
        return seq_data, data_mean, data_std

def train(model, dataset, epochs, optim, batch_size, criterion,shuffle=True):
    print('training on :', device)
    data_loader = DataLoader(dataset, batch_size = batch_size, shuffle=shuffle)
    for epoch in range(epochs):
        train_loss = 0
        model.train()
        for x, label in data_loader:
            x, label = x.permute(1,0,2).to(device), label.to(device)
            pred = model(x) 
            loss = criterion(pred, label)
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss += loss.item()
        train_loss /= len(data_loader)
        print('epoch: %d, lr: %.8f, train loss : %.8f' % (epoch + 1, scheduler.get_last_lr()[0],train_loss))
        scheduler.step() 
    pred_array, true_array = test(model, dataset_test, batch_size,criterion, shuffle=False)

    return pred_array, true_array

def test(model, dataset, batch_size, criterion, shuffle = False):
    
    model.eval()
    val_loss = 0.
    data_loader = DataLoader(dataset, batch_size, shuffle = shuffle)
    pred_list = []
    true_list = []
    
    for x, label in data_loader:
        x, label = x.permute(1,0,2).to(device), label.to(device)
        pred = model(x)
        loss = criterion(pred, label)
        val_loss += loss.item()
        pred = pred.squeeze(2).detach().cpu().numpy()
        true = label.squeeze(2).detach().cpu().numpy()
       
        pred = pred * dataset.data_std['Global_active_power'] + dataset.data_mean['Global_active_power']
        true = true * dataset.data_std['Global_active_power'] + dataset.data_mean['Global_active_power']
        x_true = x.permute(1,0,2)[:, :, -1].detach().cpu().numpy() 
        x_true = x_true * dataset.data_std['Global_active_power'] + dataset.data_mean['Global_active_power']
        combined = np.concatenate((x_true, true), axis=1)
        pred_list.append(pred)
        true_list.append(combined)
    
    pred_array = np.vstack(pred_list)
    true_array = np.vstack(true_list)  
    print(' val loss : %.8f' % (val_loss/len(data_loader)))
    return pred_array, true_array


def draw_one_sample(pred_array, true_array, i=0):
    pred = pred_array[i]
    true = true_array[i]

    historical_true = true[:seq_length]
    target_true = true[seq_length:]

    plt.figure(figsize=(8, 4))
    plt.plot(historical_true, label='Historical Data', color='blue', marker='o', markersize=3)
    plt.plot(range(seq_length, seq_length+output_length), target_true, label='Target True Values', color='green', marker='o', markersize=3)
    plt.plot(range(seq_length, seq_length+output_length), pred, label='Predicted Values', color='red', linestyle='--', marker='x', markersize=3)
    plt.legend()
    plt.title(f'Prediction vs True Values for Sample {i+1}')
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.grid()
    plt.show()

def calculate(pred_array, true_array,i):
    true_values = true_array[i, -output_length:]
    mae = np.mean(np.abs(pred_array[i,:] - true_values))
    mse = np.mean((pred_array[i,:] - true_values)**2)
    return mae,mse

seed = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
np.random.seed(seed)  
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
 
seq_length = 90
output_length = 365
features=[   
        'Global_reactive_power',
        'Sub_metering_1',
        'Sub_metering_2',
        'Sub_metering_3',
        'Sub_metering_remainder',
        'Voltage',
        'Global_intensity',
        'RR',
        'NBJRR1',
        'NBJRR5',
        'NBJRR10',
        'NBJBROU',
        'Global_active_power',]
input_size = len(features)
output_size = 1
data_path_train = pd.read_csv('train.csv')
data_path_test = pd.read_csv('test.csv')
data_path_train,data_path_test = preprocess_data(data_path_train,data_path_test)

dataset_train = get_dataset(data_path_train, seq_length, output_length, features)
dataset_test = get_dataset(data_path_test, seq_length, output_length, features)


epochs = 120
batch_size = 32
d_model = 24
nhead = 12
num_layers = 2
dropout = 0.2
pred_array_list = []
true_array_list=[]
for i in range(5):
    lr = 0.001
    model = TransformerTimeSeriesModel(input_size, output_size, seq_length, output_length, d_model, nhead,
                                   num_layers, dropout = dropout).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, 1.0, gamma=0.98)
    criterion = nn.MSELoss()
    pred_array, true_array = train(model, dataset_train, epochs, optim, batch_size, criterion, shuffle = True)
    print(pred_array.shape)
    print(true_array.shape)
    pred_array_list.append(pred_array)


np_pred_array = np.array(pred_array_list)
np_true_array=np.array(true_array)

print('pred shape:',np_pred_array.shape)
print('true shape:',np_true_array.shape)

np.save('pred_array_lstm.npy', np_pred_array)
np.save('true_lstm.npy', np_true_array)

postprocess(np_pred_array,np_true_array,model_name="transformer365",seq_length=seq_length, output_length=output_length)

print("done")