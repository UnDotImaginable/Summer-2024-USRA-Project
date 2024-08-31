import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import torch.optim as optim
import time
import os


def train_and_evaluate_lstm(input_csv, model_save_path, scaler_save_path, num_epochs=60, batch_size=32, test_size=0.2, random_state=42):
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the data
    data = pd.read_csv(input_csv)
    print('Finished loading data')

    # Extract features (X) and target (y)
    y = data[['Solar Radiation']].values  # Target for LSTM
    X = data.drop(['Solar Radiation'], axis=1).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Standardize the features using only the training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Apply the same transformation to the test data
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler for future use
    joblib.dump(scaler, scaler_save_path)

    # Reshape data for LSTM [batch_size, sequence_length, num_features]
    X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    # Convert data to PyTorch tensors and move to the device (GPU or CPU)
    X_train = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    # Create DataLoader for batch processing
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define the LSTM model
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0.3):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
            self.fc = nn.Linear(hidden_size, 1)  # Output 1 value (Wind Speed)
        
        def forward(self, x):
            h_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)
            c_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)
            out, _ = self.lstm(x, (h_0, c_0))
            out = self.fc(out[:, -1, :])  # Take output from the last time step
            return out

    input_size = X_train.shape[2]  # Number of features
    hidden_size = 64  # Number of LSTM units
    num_layers = 3  # Number of LSTM layers

    model = LSTMModel(input_size, hidden_size, num_layers).to(device)

    # Define loss function and optimizer with L2 regularization
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)

    # Train the model
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_losses.append(running_loss / len(train_loader))
        
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
        
        test_losses.append(test_loss / len(test_loader))
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}')

    # Save the model
    torch.save(model.state_dict(), model_save_path)

    # Save losses to a CSV file
    loss_df = pd.DataFrame({'epoch': range(1, num_epochs+1), 'train_loss': train_losses, 'test_loss': test_losses})
    loss_df.to_csv('training_losses_sl.csv', index=False)

    # Evaluate the model using additional metrics

    # Make predictions on the test set
    model.eval()
    y_test_pred = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            outputs = model(inputs)
            y_test_pred.extend(outputs.cpu().numpy())

    y_test_pred = np.array(y_test_pred)
    y_test_np = y_test.cpu().numpy()

    # Make predictions on the training set
    y_train_pred = []
    with torch.no_grad():
        for inputs, _ in train_loader:
            outputs = model(inputs)
            y_train_pred.extend(outputs.cpu().numpy())

    y_train_pred = np.array(y_train_pred)
    y_train_np = y_train.cpu().numpy()

    # Calculate R^2, MAE, and RMSE for the test set
    r2_test = r2_score(y_test_np, y_test_pred)
    mae_test = mean_absolute_error(y_test_np, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test_np, y_test_pred))

    # Calculate R^2, MAE, and RMSE for the training set
    r2_train = r2_score(y_train_np, y_train_pred)
    mae_train = mean_absolute_error(y_train_np, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train_np, y_train_pred))

    # Print the results
    print(f'R^2 Score (Test): {r2_test:.4f}')
    print(f'Mean Absolute Error (MAE) (Test): {mae_test:.4f}')
    print(f'Root Mean Squared Error (RMSE) (Test): {rmse_test:.4f}')
    print(f'R^2 Score (Train): {r2_train:.4f}')
    print(f'Mean Absolute Error (MAE) (Train): {mae_train:.4f}')
    print(f'Root Mean Squared Error (RMSE) (Train): {rmse_train:.4f}')

    # Save the evaluation metrics to a CSV file
    metrics_df = pd.DataFrame({
        'metric': ['R^2 Score (Test)', 'Mean Absolute Error (Test)', 'Root Mean Squared Error (Test)', 
                   'R^2 Score (Train)', 'Mean Absolute Error (Train)', 'Root Mean Squared Error (Train)'],
        'value': [r2_test, mae_test, rmse_test, r2_train, mae_train, rmse_train]
    })
    metrics_df.to_csv('evaluation_metrics_sl.csv', index=False)

    # Plot training & validation loss values
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_plot_sl.png')
    plt.show()

    
    
    
    
    
    
    
    
    
    
    
    

def predict(model_path, scaler_path, input_csv, output_csv, plot_path=None):
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define the LSTM model class
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0.3):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
            self.fc = nn.Linear(hidden_size, 1)  # Adjust the output size if you're predicting more than one target
        
        def forward(self, x):
            h_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)
            c_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)
            out, _ = self.lstm(x, (h_0, c_0))
            out = self.fc(out[:, -1, :])  # Take output from the last time step
            return out

    # Load the trained model
    input_size = 12  # Adjust according to your number of features
    hidden_size = 64  # Same as in training
    num_layers = 3  # Same as in training
    model = LSTMModel(input_size, hidden_size, num_layers).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load the scaler
    scaler = joblib.load(scaler_path)

    # Load new data for prediction
    new_data = pd.read_csv(input_csv)
    print('Finished loading new data')

    # Extract features (X) for prediction and drop 'Solar Radiation' (or relevant feature)
    X_new = new_data.drop('Solar Radiation', axis=1).values

    # Standardize the features
    X_new_scaled = scaler.transform(X_new)

    # Reshape the data for LSTM [batch_size, sequence_length, num_features]
    X_new_scaled = X_new_scaled.reshape((X_new_scaled.shape[0], 1, X_new_scaled.shape[1]))

    # Convert data to PyTorch tensors and move to the device (GPU or CPU)
    X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32).to(device)

    # Make predictions
    with torch.no_grad():
        predictions = model(X_new_tensor).cpu().numpy()

    # Save predictions to CSV
    prediction_df = pd.DataFrame(predictions, columns=['Predicted Solar Radiation'])
    prediction_df.to_csv(output_csv, index=False)

    # Plot predictions with time in days
    if plot_path:
        plt.figure(figsize=(12, 6))
        time_in_days = [i * 10 / (60 * 24) for i in range(len(predictions))]  # Convert 10-minute intervals to days
        plt.plot(time_in_days, predictions, label='Predicted Solar Radiation')
        plt.title('Predicted Solar Radiation')
        plt.xlabel('Time (Days)')
        plt.ylabel('Solar Radiation (W/m^2)')
        plt.legend()
        plt.savefig(plot_path)
        plt.show()

    print("Finished making predictions")












def find_best_week(predictions_csv, actual_csv, plot_path=None):
    # Read the predicted data
    pred_df = pd.read_csv(predictions_csv)
    pred = pred_df['Predicted Solar Radiation'].values
    pred[pred < 0] = 0  # Set everything below zero to zero

    # Read the actual data
    actual_df = pd.read_csv(actual_csv)
    actual = actual_df['Solar Radiation'].values

    # Extract date information
    years = actual_df['year'].values
    days_of_year = actual_df['day_of_year'].values
    times = actual_df['time'].values

    # Ensure both arrays have the same length
    min_length = min(len(pred), len(actual))
    pred = pred[:min_length]
    actual = actual[:min_length]
    years = years[:min_length]
    days_of_year = days_of_year[:min_length]
    times = times[:min_length]

    r2_array = []
    # Split pred and actual by weeks (24*7)
    for i in range(0, len(pred), 24*7):
        pred_week = pred[i:i+24*7]
        actual_week = actual[i:i+24*7]
        # Calculate the mean of the actual values
        mean_actual = np.mean(actual_week)

        # Calculate the total sum of squares
        ss_total = np.sum((actual_week - mean_actual) ** 2)

        # Calculate the residual sum of squares
        ss_res = np.sum((actual_week - pred_week) ** 2)

        # Calculate the R-squared value
        r2 = 1 - (ss_res / ss_total)
        r2_array.append(r2)

    # Find the week with the maximum R-squared value
    max_r2 = max(r2_array)
    index = r2_array.index(max_r2)
    print(f'Max R-squared: {max_r2:.4f}')
    print(f'Index: {index}')

    # Calculate RMSE for that week
    rmse = np.sqrt(np.mean((actual[index*24*7:index*24*7+24*7] - pred[index*24*7:index*24*7+24*7]) ** 2))
    print(f'RMSE: {rmse:.4f}')

    # Calculate MAE for that week
    mae = np.mean(np.abs(actual[index*24*7:index*24*7+24*7] - pred[index*24*7:index*24*7+24*7]))
    print(f'Mean Absolute Error: {mae:.4f}')

    # Get the dates for the best week
    year_week = years[index*24*7:index*24*7+24*7]
    day_of_year_week = days_of_year[index*24*7:index*24*7+24*7]
    time_week = times[index*24*7:index*24*7+24*7]

    # Create a list of datetime strings for the best week
    date_labels = [f"{year}-{day_of_year} {time}" for year, day_of_year, time in zip(year_week, day_of_year_week, time_week)]

    # Plot the week for actual and prediction with that index
    pred_week = pred[index*24*7:index*24*7+24*7]
    actual_week = actual[index*24*7:index*24*7+24*7]

    plt.figure(figsize=(12, 6))
    plt.plot(actual_week, label='Actual Solar Radiation')
    plt.plot(pred_week, label='Predicted Solar Radiation')
    plt.title(f'Actual vs Predicted Solar Radiation - Best Week\nR-squared: {max_r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}')
    plt.xlabel('Time')
    plt.xticks(ticks=range(0, len(date_labels), 24), labels=date_labels[::24], rotation=45)
    plt.ylabel('Solar Radiation')
    plt.legend()

    if plot_path:
        plt.savefig(plot_path)
    
    plt.show()

    return max_r2, rmse, mae

