import torch
import torch.nn as nn
import torch.optim as optim
import os

# Define the model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Get the output from the last time step
        out = self.dropout(lstm_out)
        out = self.fc(out)
        return out

def lstm_train(x_train, y_train, update=False):
    print('Training model...')

    # Device configuration (move model and data to GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Assuming input data (x_train) has dimensions (batch_size, seq_len, input_size)
    input_size = x_train.shape[2]  # Number of features per time step
    seq_len = x_train.shape[1]  # Number of time steps
    hidden_size = 30
    output_size = 1
    dropout_prob = 0.2

    model = LSTMModel(input_size=input_size, hidden_size=hidden_size, 
                            output_size=output_size, dropout_prob=dropout_prob).to(device)
    
    if os.path.exists(f'models/lstm_model.pth') and update:

        # Load the saved state_dict into the model
        model.load_state_dict(torch.load(f'models/lstm_model.pth', weights_only=True))


    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    # Convert numpy arrays to PyTorch tensors and move to device
    inputs = torch.tensor(x_train, dtype=torch.float32).to(device)
    labels = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)  # Ensure labels are shaped (batch_size, 1)

    # Training loop
    epochs = 20
    for epoch in range(epochs):
        model.train()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss and other metrics if needed
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    # Save the model's state_dict to a file
    model_path = f'models/lstm_model.pth'
    torch.save(model.state_dict(), model_path)
    print("Model trained and saved!")
    return model_path

def lstm_test(x_test, y_test, model_path=f'models/lstm_model.pth'):

    # Device configuration (move model and data to GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_size = x_test.shape[2]  # Number of features per time step
    seq_len = x_test.shape[1]  # Number of time steps
    hidden_size = 30
    output_size = 1
    dropout_prob = 0.2

    # Recreate the model architecture (same as before)
    model = LSTMModel(input_size=input_size, hidden_size=hidden_size, 
                            output_size=output_size, dropout_prob=dropout_prob).to(device)

    # Load the saved state_dict into the model
    model.load_state_dict(torch.load(model_path, weights_only=True))

    # Set the model to evaluation mode if you're using it for inference
    model.eval()
    criterion = nn.L1Loss()

    # Convert test data to PyTorch tensors and move to device
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)  # Ensure correct shape

    # Disable gradient calculation (for faster inference)
    with torch.no_grad():
        # Forward pass: Get predictions
        test_outputs = model(x_test_tensor)
        
        # Calculate test loss
        test_loss = criterion(test_outputs, y_test_tensor)

    # Print the test loss
    print(f'Test Loss: {test_loss.item():.4f}')