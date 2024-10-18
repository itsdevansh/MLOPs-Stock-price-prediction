from data_handling import data_extract, data_processing
from model import lstm_train, lstm_test

def train(ticker, period):# Train new model
    data = data_extract(start_date=None, end_date=None, period=period, ticker=ticker)
    x_train, y_train, x_test, y_test = data_processing(data)
    model = lstm_train(x_train, y_train)
    lstm_test(x_test, y_test, model)


def update(ticker, start_date, end_date): # Update existing model
    data = data_extract(start_date=start_date, end_date=end_date, ticker=ticker)
    x_train, y_train, _, _ = data_processing(data, train_split=1)
    model = lstm_train(x_train, y_train, update=True, epochs=1)

train('MSFT', 'max')

# update('MSFT', 'start date','end date')