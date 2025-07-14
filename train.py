





def train_model(model, train_loader, criterion=nn.MSELoss(), epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step(loss)

def evaluate(model, test_loader):
    model.eval()
    with torch.no_grad():
        predictions, truths = [], []
        for inputs, targets in test_loader:
            outputs = model(inputs)
            predictions.extend(outputs.numpy())
            truths.extend(targets.numpy())
        return {
            'MSE': mean_squared_error(truths, predictions),
            'MAE': mean_absolute_error(truths, predictions)
        }