def calculate_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Calculate Stochastic Oscillator
def calculate_stochastic(df, period=14):
    lowest_low = df['Low'].rolling(window=period).min()
    highest_high = df['High'].rolling(window=period).max()
    
    stoch_k = ((df['Close'] - lowest_low) / (highest_high - lowest_low)) * 100
    return stoch_k

# Calculate Williams %R
def calculate_williams_r(df, period=14):
    highest_high = df['High'].rolling(window=period).max()
    lowest_low = df['Low'].rolling(window=period).min()
    
    williams_r = ((highest_high - df['Close']) / (highest_high - lowest_low)) * -100
    return williams_r

def calculate_moving_averages(df):
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA100'] = df['Close'].rolling(window=100).mean()
    return df

def calculate_price_returns(df):
    df['Returns'] = df['Close'].pct_change()
    return df

def lagged_features(df):
    for i in range(1, 6):
        df[f'Close_lag_{i}'] = df['Close'].shift(i)
    return df
def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    df['EMA_short'] = df['Close'].ewm(span=short_window, adjust=False).mean()
    df['EMA_long'] = df['Close'].ewm(span=long_window, adjust=False).mean()
    df['MACD'] = df['EMA_short'] - df['EMA_long']
    df['MACD_Signal'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()
    return df

def calculate_bollinger_bands(df, window=20, num_std_dev=2):
    df['BB_Middle'] = df['Close'].rolling(window=window).mean()
    df['BB_Std_Dev'] = df['Close'].rolling(window=window).std()
    df['BB_Upper'] = df['BB_Middle'] + (num_std_dev * df['BB_Std_Dev'])
    df['BB_Lower'] = df['BB_Middle'] - (num_std_dev * df['BB_Std_Dev'])
    return df

def calculate_ichimoku(df):
    df['tenkan_sen'] = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2
    df['kijun_sen'] = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
    df['senkou_span_b'] = ((df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2).shift(26)
    df['chikou_span'] = df['Close'].shift(-26)
    return df

def calculate_technical_indicators(df):
    df['RSI'] = calculate_rsi(df)
    df['%K'] = calculate_stochastic(df)
    df['%R'] = calculate_williams_r(df)
    df = calculate_moving_averages(df)
    df = calculate_price_returns(df)
    df = lagged_features(df)
    df = calculate_macd(df)
    df = calculate_bollinger_bands(df)
    df = calculate_ichimoku(df)
    return df