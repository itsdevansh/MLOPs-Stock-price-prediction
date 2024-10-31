import logging
import pandas as pd
from zenml import step
import yfinance as yf

class IngestData:
    def __init__(self, ticker: str, period: str) -> None:
        self.ticker = ticker
        self.period = period

    def get_data(self):
        logging.info(f"Ingesting stock data of stock {self.ticker}")
        data = pd.DataFrame(yf.download(self.ticker, period=self.period))
        return data
    
@step
def ingest_df(ticker: str) -> pd.DataFrame:

    try:
        ingest_data = IngestData(ticker, period='max')
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error while ingesting data: {e}")
        raise e