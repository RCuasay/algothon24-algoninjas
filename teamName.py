import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

nInst = 50
currentPos = np.zeros(nInst)

def calculate_features(prcSoFar):
    (nins, nt) = prcSoFar.shape
    features = []
    for i in range(nins):
        prices = prcSoFar[i, :]
        ma7 = pd.Series(prices).rolling(window=7).mean().values
        ma21 = pd.Series(prices).rolling(window=21).mean().values
        ma50 = pd.Series(prices).rolling(window=50).mean().values
        ma100 = pd.Series(prices).rolling(window=100).mean().values

        # Improved RSI calculation with error handling
        diff = pd.Series(prices).diff()
        gain = diff.where(diff > 0, 0).rolling(window=14).mean()
        loss = -diff.where(diff < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        volatility = pd.Series(prices).rolling(window=7).std().values
        momentum = pd.Series(prices).diff(3).values

        # Combine all features
        features.append(np.column_stack((ma7, ma21, ma50, ma100, rsi, volatility, momentum)))
    return np.array(features).reshape(nins, nt, -1)

def train_model(prcSoFar):
    features = calculate_features(prcSoFar)
    target = np.log(prcSoFar[:, 1:] / prcSoFar[:, :-1])

    X = features[:, :-1, :].reshape(-1, features.shape[2])
    y = target.flatten()

    # Use SimpleImputer to handle NaNs
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    return model, imputer

def predict_future_prices(model, imputer, prcSoFar):
    features = calculate_features(prcSoFar)
    current_features = features[:, -1, :]
    current_features = imputer.transform(current_features)
    predictions = model.predict(current_features)
    return np.exp(predictions) * prcSoFar[:, -1]

def trading_strategy(predictions, prcSoFar, max_daily_trade_value=10000):
    global currentPos
    daily_trades = np.zeros(nInst)
    total_value = 0

    for i in range(nInst):
        predicted_price = predictions[i]
        current_price = prcSoFar[i, -1]

        # Buy signal
        if predicted_price > current_price:
            trade_value = min(max_daily_trade_value - total_value, 1000)
            if trade_value > 0:
                daily_trades[i] = trade_value / current_price
                total_value += trade_value

        # Sell signal (optional: implement better sell strategy)
        elif predicted_price < current_price:
            trade_value = min(max_daily_trade_value - total_value, 1000)
            if trade_value > 0:
                daily_trades[i] = -trade_value / current_price
                total_value += trade_value

    currentPos += daily_trades
    return currentPos

# Train the model initially
model = None
imputer = None

def getMyPosition(prcSoFar):
    global model, imputer, currentPos

    (nins, nt) = prcSoFar.shape
    if nt < 100:  # Ensure we have enough data for features
        return np.zeros(nins)

    if model is None:
        model, imputer = train_model(prcSoFar)

    predictions = predict_future_prices(model, imputer, prcSoFar)
    currentPos = trading_strategy(predictions, prcSoFar)

    return currentPos
