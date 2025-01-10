import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
from zenml.client import Client
from zenml.integrations.mlflow.services.mlflow_deployment import MLFlowDeploymentService

# Import your pipeline running functions
from pipelines.training_pipeline import train_pipeline
from pipelines.deployment_pipeline import continuous_deployment_pipeline

# Functions to fetch and visualize data
def fetch_stock_data(ticker, period='1y'):
    import yfinance as yf
    df = yf.download(ticker, period=period)
    return df

def visualize_stock_data(df):
    st.line_chart(df['Close'])

# Main Streamlit App
st.title("Stock Price Prediction with LSTM")

# Sidebar options
option = st.sidebar.selectbox(
    "Choose an Action",
    ["Home", "View Stock Data", "Train Model", "Deploy Model", "Batch Prediction"]
)

if option == "Home":
    st.write("Welcome to the Stock Price Prediction App!")
    st.write("Choose an option from the sidebar to get started.")

elif option == "View Stock Data":
    st.header("View Stock Data")
    ticker = st.text_input("Enter Stock Ticker (e.g., MSFT, AAPL)", value="MSFT")
    period = st.selectbox("Select Time Period", ["1mo", "3mo", "6mo", "1y", "max"], index=4)
    
    if st.button("Fetch Data"):
        try:
            data = fetch_stock_data(ticker, period)
            st.write(data.tail())
            st.write("### Stock Closing Price")
            visualize_stock_data(data)
        except Exception as e:
            st.error(f"Error fetching data: {e}")

elif option == "Train Model":
    st.header("Train the Model")
    ticker = st.text_input("Enter Stock Ticker for Training", value="MSFT")
    
    if st.button("Run Training Pipeline"):
        try:
            train_pipeline(ticker=ticker)
            st.success("Training completed successfully!")
        except Exception as e:
            st.error(f"Error during training: {e}")

elif option == "Deploy Model":
    st.header("Deploy the Model")
    ticker = st.text_input("Enter Stock Ticker for Deployment", value="MSFT")
    
    if st.button("Run Deployment Pipeline"):
        try:
            continuous_deployment_pipeline(ticker=ticker)
            st.success("Deployment completed successfully!")
        except Exception as e:
            st.error(f"Error during deployment: {e}")

elif option == "Batch Prediction":
    st.header("Batch Prediction")
    
    # Fetch the active model deployment service
    client = Client()
    model_deployer = client.active_stack.model_deployer
    deployed_services = model_deployer.find_model_server(
        pipeline_name="continuous_deployment_pipeline",  # Name of the deployment pipeline
        step_name="mlflow_model_deployer_step",          # Name of the deployment step
        model_name="lstm_alpha",                         # Name of the registered model
        running=True                                     # Ensure the service is running
    )

    if deployed_services:
        service = deployed_services[0]  # Assuming only one deployed service
        st.write(f"Prediction service is running at: {service.prediction_url}")

        input_data = st.file_uploader("Upload Batch Data (CSV)", type=["csv"])
        if input_data is not None:
            try:
                df = pd.read_csv(input_data)
                st.write("### Uploaded Data", df.head())

                # Send data to the prediction endpoint
                response = requests.post(
                    service.prediction_url, 
                    json={"data": df.to_dict(orient="records")}
                )
                if response.status_code == 200:
                    predictions = response.json()
                    st.write("### Predictions", predictions)
                else:
                    st.error(f"Prediction failed: {response.text}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
    else:
        st.warning("No active deployed model found. Please deploy a model first.")

