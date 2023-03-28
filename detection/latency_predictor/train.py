import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset 

from latency_pred_model import load_latency_data, save_model, LatencyPredictor, AverageMeter

path = 'LUT.csv'
lut_data = load_latency_data(path)

train, test = train_test_split(lut_data, test_size=0.2, random_state=42, shuffle=True)

X_train, X_test = train.values[:, :-1], test.values[:, :-1]
y_train, y_test = train.values[:, -1], test.values[:, -1]

train_dataset = TensorDataset(torch.tensor(X_train).float(), 
                                    torch.tensor(y_train).float())
test_dataset = TensorDataset(torch.tensor(X_test).float(), 
                                    torch.tensor(y_test).float())

train = torch.utils.data.DataLoader(
            train_dataset, batch_size=8, shuffle=True
            )
test = torch.utils.data.DataLoader(
            test_dataset, batch_size=8, shuffle=False
)

in_features = lut_data.shape[1] - 1
h_units = 256
epochs = 200


model = LatencyPredictor(in_features, h_units)
model = model.cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_loss = []
test_loss = []
best_loss = 0
for step, epoch in tqdm(enumerate(range(1, epochs+1))):
    # Train
    model.train()
    train_losses = AverageMeter()
    for i, data in enumerate(train):
        inputs, targets = data
        inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        train_losses.update(loss.item(), inputs.size(0))
    train_loss.append(train_losses.avg)
        # if step % 20 == 0:
        #     print(f'Train loss')
    
    # Eval
    model.eval()
    test_losses = AverageMeter()
    for i, data in enumerate(test):
        inputs, targets = data
        inputs, targets = inputs.cuda(), targets.cuda()

        with torch.no_grad():
            output = model(inputs)
            loss = criterion(output, targets)
        test_losses.update(loss.item(), inputs.size(0))
    test_loss.append(test_losses.avg)

    if best_loss < test_losses.avg:
        best_loss = test_losses.avg
        save_model(epoch, model, optimizer, criterion)
    print(f'Epoch[{epoch}] Train loss: {train_losses.avg:.3f}, Test loss: {test_losses.avg:.3f}')

loss_data = {
    'Train Loss': train_loss, 
    'Test Loss': test_loss
}
loss_df = pd.DataFrame(loss_data)

load_state = torch.load('best_model.pth')
model.load_state_dict(load_state['model_state_dict'])

model.eval()
test_preds = []
for i, data in enumerate(test):
    inputs, targets = data
    inputs, targets = inputs.cuda(), targets.cuda()

    with torch.no_grad():
        pred = model(inputs)
    test_preds.extend(pred.cpu().numpy())
    
data = pd.DataFrame(
        {'Latency Prediction': np.array(test_preds).reshape(-1), 
        'Real Latency': y_test}
    )
print(f"Prediction & Latency Correlation : {data.corr().iloc[0, 1]:3f}")