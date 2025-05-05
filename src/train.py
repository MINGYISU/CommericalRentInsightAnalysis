import torch.nn as nn
import torch.nn.functional as F
import joblib
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from utils import *
from model import LeaseModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = LeaseModel(len(features), 1).to(device)

print('Using device:', device)
# Print the 
print('model architecture:\n' + str(model))
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params / 1e6:.2f}M")

if os.path.exists(data_path) and os.path.exists(scaler_path):
    print('Processed data found. Loading ... ...')
    data = torch.load(data_path)
    X_train = data['X_train'].to(device)
    y_train = data['y_train'].to(device)
    X_test = data['X_test'].to(device)
    y_test = data['y_test'].to(device)
    scalers = joblib.load(scaler_path)
    feature_scaler = scalers['feature_scaler']
    target_scaler = scalers['target_scaler']
    print("Processed data loaded successfully.")
else:
    print('Processed data not found. Processing from source ... ...')
    csv_file = 'data/cleaned_leases_features.csv'
    df = pd.read_csv(csv_file)

    X = df.drop(['rent', 'leasedSF_log'], axis=1)
    y = df['rent']

    X = X.apply(get_mapping, axis=1)
    X = X.to_numpy(np.float32)
    y = y.to_numpy(np.float32)

    # Split the data first
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.2, random_state=49)

    # Feature normalization
    feature_scaler = StandardScaler()
    X_train = feature_scaler.fit_transform(X_train_raw)
    X_test = feature_scaler.transform(X_test_raw)  # Use the same scaler for test data

    # Target normalization
    target_scaler = StandardScaler()
    y_train = target_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
    y_test = target_scaler.transform(y_test_raw.reshape(-1, 1)).flatten()

    # Convert to PyTorch tensors
    X_train = torch.from_numpy(X_train.astype(np.float32)).to(device)
    y_train = torch.from_numpy(y_train.astype(np.float32)).to(device)
    X_test = torch.from_numpy(X_test.astype(np.float32)).to(device)
    y_test = torch.from_numpy(y_test.astype(np.float32)).to(device)

    # Save the processed data
    torch.save({
        'X_train': X_train.cpu(),
        'X_test': X_test.cpu(),
        'y_train': y_train.cpu(),
        'y_test': y_test.cpu(),
    }, data_path)

    joblib.dump({
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler
    }, scaler_path)

    print(f"Data processed and saved to {data_path}.")
    print(f"Scalers saved to {scaler_path}.")

print(f"Feature mean: {feature_scaler.mean_}")
print(f"Feature std: {feature_scaler.scale_}")
print(f"Target mean: {target_scaler.mean_[0]:.4f}, Target std: {target_scaler.scale_[0]:.4f}")

batch_size = 256
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 1000
criterion = nn.MSELoss()
losses = []

# Update the training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        # Reshape outputs and targets for MSELoss
        outputs = outputs.squeeze()  # Remove extra dimension if needed
        batch_y = batch_y.float()    # Ensure targets are float
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_X.size(0)
    
    epoch_loss = running_loss / len(train_dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    losses.append(epoch_loss)

torch.save({
    'model_state_dict': model.state_dict(), 
    'features_order': features, 
    'mapping': mapping}, model_path)

# graph it
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), losses)
plt.ylabel('Mean Squared Error')
plt.title('Training Loss')
plt.xlabel('Epoches')
plt.savefig(os.path.join('training_log', f'training_loss_{len(os.listdir("training_log")) // 2}.png'), dpi=300, bbox_inches='tight')

# Update the evaluation section for regression
from sklearn.metrics import r2_score

model.eval()
with torch.no_grad():
    test_loss = 0.0
    mae = 0.0  # Mean Absolute Error
    all_preds = []
    all_targets = []
    
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        outputs = outputs.squeeze()
        batch_y = batch_y.float()
        
        # Store predictions and targets for R² calculation
        all_preds.extend(outputs.cpu().numpy())
        all_targets.extend(batch_y.cpu().numpy())
            
        # Calculate MSE loss
        loss = criterion(outputs, batch_y)
        test_loss += loss.item() * batch_X.size(0)
            
        # Calculate MAE
        mae += torch.abs(outputs - batch_y).sum().item()
    
    # Calculate metrics on normalized data
    avg_test_loss = test_loss / len(test_dataset)
    rmse = np.sqrt(avg_test_loss)
    avg_mae = mae / len(test_dataset)
    r2 = r2_score(all_targets, all_preds)
    
    print(f'Normalized Test MSE: {avg_test_loss:.4f}')
    print(f'Normalized Test RMSE: {rmse:.4f}')
    print(f'Normalized Test MAE: {avg_mae:.4f}')
    print(f'R² Score: {r2:.4f}')
    
    # Calculate mean of normalized target variable
    mean_target = np.mean(all_targets)
    print(f'Normalized Mean Target: {mean_target:.4f}')
    print(f'Normalized RMSE as % of Mean: {(rmse/mean_target)*100:.2f}%')
    
    # Convert predictions and targets back to original scale for interpretability
    denorm_preds = target_scaler.inverse_transform(np.array(all_preds).reshape(-1, 1)).flatten()
    denorm_targets = target_scaler.inverse_transform(np.array(all_targets).reshape(-1, 1)).flatten()
    
    # Calculate metrics in original scale
    denorm_mse = np.mean((denorm_preds - denorm_targets) ** 2)
    denorm_rmse = np.sqrt(denorm_mse)
    denorm_mae = np.mean(np.abs(denorm_preds - denorm_targets))
    denorm_r2 = r2_score(denorm_targets, denorm_preds)
    
    print("\nMetrics in original scale:")
    print(f'Test MSE: {denorm_mse:.4f}')
    print(f'Test RMSE: {denorm_rmse:.4f}')
    print(f'Test MAE: {denorm_mae:.4f}')
    print(f'R² Score: {denorm_r2:.4f}')
    
    # Calculate mean of original target variable
    denorm_mean_target = np.mean(denorm_targets)
    print(f'Mean Target: {denorm_mean_target:.4f}')
    print(f'RMSE as % of Mean: {(denorm_rmse/denorm_mean_target)*100:.2f}%')
    
    # Plot predictions vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(denorm_targets, denorm_preds, alpha=0.3)
    plt.plot([min(denorm_targets), max(denorm_targets)], [min(denorm_targets), max(denorm_targets)], 'r--')
    plt.xlabel('Actual Rent')
    plt.ylabel('Predicted Rent')
    plt.title('Predicted vs Actual Rent Values')
    plt.savefig(os.path.join('training_log', f'predictions_{len(os.listdir("training_log")) // 2}.png'), dpi=300, bbox_inches='tight')
