import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from preprocess import preprocess_data
from lstm import LSTMModel_S, LSTMModel_L
from postprocess import postprocess

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

train_data, test_data = preprocess_data(train_data, test_data)

input_dim = 13
hidden_dim = 96
input_seq = 90
output_seq = 365
layer_dim = 2  # Only one layer
output_dim = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

features=[
        'Global_active_power',
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
        'NBJBROU']

tgt='Global_active_power'

x_train=train_data[features]
y_train=train_data[tgt]

x_test=test_data[features]
y_test=test_data[tgt]

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

def create_sequences(x, y, time_steps=90, seq_length=90):
    xs, ys = [], []
    for i in range(len(x) - time_steps - seq_length):
        xs.append(x[i:(i + time_steps), :])
        ys.append(y[i + time_steps: i + time_steps + seq_length])
    return np.array(xs), np.array(ys)

# Create sequences before splitting the data
x_train_seq, y_train_seq = create_sequences(x_train_scaled, y_train, input_seq, output_seq)
x_test_seq, y_test_seq = create_sequences(x_test_scaled, y_test, input_seq, output_seq)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(x_train_seq, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32).view(-1, output_seq).to(device)
X_test_tensor = torch.tensor(x_test_seq, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test_seq, dtype=torch.float32).view(-1, output_seq).to(device)
# Create TensorDataset for training and validation
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
# DataLoader for batch training
batch_size = 32
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def test(model):
    
    model.eval()
    pred_list = []
    true_list = []
    with torch.no_grad():
        for x,label in test_loader:
            y = model(x).squeeze(-1)
            for row in y.detach().cpu().numpy():
                pred_list.append(list(row))
            for row in label.detach().cpu().numpy():
                true_list.append(list(row))
    return pred_list, true_list

def train(flag,i):
    if flag=='s':
        output_seq=90
        lstm_model_torch = LSTMModel_S(input_dim, hidden_dim, layer_dim, output_dim, input_seq, output_seq).to(device)
    elif flag=='l':
        output_seq=365
        lstm_model_torch = LSTMModel_L(input_dim, hidden_dim, layer_dim, output_dim, input_seq, output_seq).to(device)
    criterion_mse = nn.MSELoss()
    optimizer = optim.Adam(lstm_model_torch.parameters(), lr=0.1)
    Loss = []
    num_epochs = 100
    for epoch in range(num_epochs):
        epoch_loss = 0
        for x,label in train_loader:
            x = x.to(device)
            outputs = lstm_model_torch(x).squeeze(-1)
            loss = criterion_mse(outputs, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Loss.append(loss.item())
            epoch_loss += loss.item()  
        print(f'Epoch {epoch} Loss: {epoch_loss/len(train_loader):.4f}')
    torch.save(lstm_model_torch.state_dict(), 'lstm_model_torch'+str(flag)+str(i)+'.pth')
    pred, true = test(lstm_model_torch)
    return pred, true


pred_array = []
true_array=[]
for i in range(5):
    pred,true = train('l',i)
    pred_array.append(pred)


np_pred_array = np.array(pred_array)
np_true_array = np.array(true)

print('pred shape:',np_pred_array.shape)
print('true shape:',np_true_array.shape)

np.save('pred_array_lstm.npy', np_pred_array)
np.save('true_lstm.npy', np_true_array)

postprocess(np_pred_array,np_true_array,model_name="lstm_l",seq_length=input_seq, output_length=output_seq)


