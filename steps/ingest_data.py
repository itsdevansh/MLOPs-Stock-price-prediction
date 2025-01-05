import logging
import pandas as pd
from zenml import step
import yfinance as yf
from datetime import datetime
import os

class IngestData:
    def __init__(self, ticker: str) -> None:
        self.ticker = ticker

    def get_data(self) -> tuple[pd.DataFrame, str]:
        action = "train"
        logging.info(f"Ingesting stock data of stock {self.ticker}")
        data_path = f"/Users/devanshk/Desktop/Stock-price-prediction{self.ticker}.csv"
        if not os.path.exists(data_path):
            data = pd.DataFrame(yf.download(self.ticker, period='max'))
            data.reset_index()
            # data['Date'] = pd.to_datetime #TODO
            data.to_csv(data_path)
        else:
            today_date = datetime.now().strftime('%Y-%m-%d')
            data = pd.read_csv(data_path)
            latest_date = data.iloc[-1]["Date"]
            today_date = datetime.strptime(today_date, "%Y-%m-%d").date()
            latest_date = datetime.strptime(latest_date, "%Y-%m-%d").date()
            if today_date > latest_date:
                new_data = pd.DataFrame(yf.download(self.ticker, start=str(latest_date), end=str(today_date))).reset_index()
                if len(new_data != 0):
                    action = "update"
                    print(new_data)
                    data = pd.concat([data, new_data])
                    data.to_csv(data_path)
                else:
                    action = "nothing"
            elif today_date == latest_date:
                action = "nothing"
        return data, action
    
@step(enable_cache=False)
def ingest_df(ticker: str) -> tuple[pd.DataFrame, str, str]:

    try:
        ingest_data = IngestData(ticker)
        df, action = ingest_data.get_data()
        return df, ticker, action
    except Exception as e:
        logging.error(f"Error while ingesting data: {e}")
        raise e