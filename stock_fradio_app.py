import gradio as gr
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class StockPricePredictor:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.dates = None  # Initialize dates attribute
        
    def load_and_preprocess_data(self, df):
        """Load and preprocess the data, including proper date handling"""
        # Convert date to datetime, handling various formats
        df['Date'] = pd.to_datetime(df['Date'], format='mixed')
        df['Date'] = pd.to_datetime(df['Date']).dt.date  # Extract just the date part
        
        # Sort by date
        df = df.sort_values('Date')
        
        # Store dates for plotting
        self.dates = df['Date'].values
        
        # Create features
        df['Returns'] = df['Close'].pct_change()
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Update dates after dropping NaN values
        self.dates = df['Date'].values
        
        # Select features for the model
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                   'Returns', 'MA5', 'MA20', 'MA50', 'Volatility']
        
        # Scale the features
        self.data = df[features].values
        self.scaled_data = self.scaler.fit_transform(self.data)
        
        return df
    
    def create_sequences(self):
        """Create sequences for LSTM training"""
        X = []
        y = []
        
        for i in range(self.sequence_length, len(self.scaled_data)):
            X.append(self.scaled_data[i-self.sequence_length:i])
            y.append(self.scaled_data[i, 3])  # Close price is at index 3
            
        X = np.array(X)
        y = np.array(y)
        
        # Split into train and test sets (80-20 split)
        train_size = int(len(X) * 0.8)
        self.X_train = X[:train_size]
        self.X_test = X[train_size:]
        self.y_train = y[:train_size]
        self.y_test = y[train_size:]
        
        # Store split indices for later use
        self.train_size = train_size
    
    def build_model(self):
        """Build the LSTM model"""
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.sequence_length, self.data.shape[1])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return self.model
    
    def train_model(self, epochs=50, batch_size=32):
        """Train the model with given parameters"""
        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1
        )
        return history
    
    def evaluate_model(self):
        """Evaluate the model and prepare results"""
        # Make predictions
        train_predictions = self.model.predict(self.X_train)
        test_predictions = self.model.predict(self.X_test)
        
        # Prepare data for inverse transformation
        train_pred_full = np.zeros((len(train_predictions), self.data.shape[1]))
        test_pred_full = np.zeros((len(test_predictions), self.data.shape[1]))
        
        train_pred_full[:, 3] = train_predictions.flatten()
        test_pred_full[:, 3] = test_predictions.flatten()
        
        # Inverse transform predictions
        train_predictions = self.scaler.inverse_transform(train_pred_full)[:, 3]
        test_predictions = self.scaler.inverse_transform(test_pred_full)[:, 3]
        
        # Get actual values
        train_actual = self.scaler.inverse_transform(self.scaled_data)[self.sequence_length:len(self.X_train)+self.sequence_length, 3]
        test_actual = self.scaler.inverse_transform(self.scaled_data)[len(self.X_train)+self.sequence_length:, 3]
        
        # Get corresponding dates
        train_dates = self.dates[self.sequence_length:len(self.X_train)+self.sequence_length]
        test_dates = self.dates[len(self.X_train)+self.sequence_length:]
        
        # Calculate RMSE
        train_rmse = np.sqrt(np.mean((train_predictions - train_actual) ** 2))
        test_rmse = np.sqrt(np.mean((test_predictions - test_actual) ** 2))
        
        return {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_predictions': train_predictions,
            'test_predictions': test_predictions,
            'train_actual': train_actual,
            'test_actual': test_actual,
            'train_dates': train_dates,
            'test_dates': test_dates
        }

def create_plotly_visualization(results):
    """Create an interactive Plotly visualization"""
    fig = make_subplots(rows=1, cols=1, 
                       subplot_titles=('Stock Price Predictions',))

    # Add traces for stock prices with dates
    fig.add_trace(
        go.Scatter(x=results['train_dates'], y=results['train_actual'], 
                  name="Training Actual", line=dict(color='blue')))
    fig.add_trace(
        go.Scatter(x=results['train_dates'], y=results['train_predictions'], 
                  name="Training Predictions", line=dict(color='lightblue')))
    
    # Add test data
    fig.add_trace(
        go.Scatter(x=results['test_dates'], y=results['test_actual'], 
                  name="Testing Actual", line=dict(color='red')))
    fig.add_trace(
        go.Scatter(x=results['test_dates'], y=results['test_predictions'], 
                  name="Testing Predictions", line=dict(color='orange')))

    # Update layout
    fig.update_layout(
        title='MSFT Stock Price Analysis',
        height=600,
        showlegend=True,
        template='plotly_white',
        xaxis_title="Date",
        yaxis_title="Stock Price ($)"
    )

    return fig

def train_and_visualize(file, epochs, sequence_length, batch_size):
    """Main function to train model and create visualization"""
    try:
        # Read the CSV file
        df = pd.read_csv(file.name)
        
        # Initialize predictor
        predictor = StockPricePredictor(sequence_length=sequence_length)
        
        # Process data
        df = predictor.load_and_preprocess_data(df)
        predictor.create_sequences()
        
        # Build and train model
        predictor.build_model()
        history = predictor.train_model(epochs=epochs, batch_size=batch_size)
        
        # Evaluate model
        results = predictor.evaluate_model()
        
        # Create visualization
        fig = create_plotly_visualization(results)
        
        # Create results summary
        summary = f"""
        Model Performance Summary:
        Training RMSE: ${results['train_rmse']:.2f}
        Testing RMSE: ${results['test_rmse']:.2f}
        """
        
        return fig, summary
        
    except Exception as e:
        return None, f"Error: {str(e)}"

# Create Gradio interface
iface = gr.Interface(
    fn=train_and_visualize,
    inputs=[
        gr.File(label="Upload CSV file"),
        gr.Slider(minimum=10, maximum=100, step=10, value=50, label="Number of Epochs"),
        gr.Slider(minimum=30, maximum=100, step=10, value=60, label="Sequence Length"),
        gr.Slider(minimum=16, maximum=64, step=16, value=32, label="Batch Size")
    ],
    outputs=[
        gr.Plot(label="Stock Price Predictions"),
        gr.Textbox(label="Model Performance Summary")
    ],
    title="MSFT Stock Price Prediction Dashboard",
    description="""
    Upload a stock price CSV file to train and visualize LSTM predictions. 
    The CSV should contain columns: Date, Open, High, Low, Close, Volume.
    """,
    theme="default"
)

if __name__ == "__main__":
    iface.launch()