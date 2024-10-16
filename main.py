from data_handling import data_extract, data_processing
from model import lstm_train, lstm_test

data = data_extract(start_date=None, end_date=None, period='max', ticker='MSFT')
x_train, y_train, x_test, y_test = data_processing(data)

model = lstm_train(x_train, y_train)
lstm_test(x_test, y_test, model)