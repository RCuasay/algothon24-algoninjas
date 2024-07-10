import numpy as np
import pandas as pd

nInst = 50
currentPos = np.zeros(nInst)

def calculate_atr(prcSoFar, window=30):
    high = prcSoFar.max(axis=1)
    low = prcSoFar.min(axis=1)
    close = prcSoFar[:, -1]
    
    tr = np.maximum(high - low, np.maximum(abs(high - close), abs(low - close)))
    atr = pd.Series(tr).rolling(window=window, min_periods=1).mean().values
    return atr

def calculate_rsi(prcSoFar, window=30):
    delta = np.diff(prcSoFar, axis=1)
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)
    
    # Transpose before rolling, then transpose back
    avg_gain = pd.DataFrame(gain).rolling(window=window).mean().to_numpy()
    avg_loss = pd.DataFrame(loss).rolling(window=window).mean().to_numpy()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return np.hstack([np.zeros((nInst, 1)), rsi])  # Pad with zeros for alignment

def getMyPosition(prcSoFar):
    global currentPos, initialPrices
    
    (nins, nt) = prcSoFar.shape
    if nt < 2:
        return np.zeros(nins)
    
    initialPrices = prcSoFar[:, 0]
    
    prcSoFar_df = pd.DataFrame(prcSoFar)
    ema = prcSoFar_df.ewm(span=30, adjust=False).mean()
    tradingPositions = (prcSoFar_df - ema).apply(np.sign)
    latestTradingPositions = tradingPositions.iloc[:, -1].to_numpy()
    
    # Calculate ATR for volatility-based position sizing
    atr = calculate_atr(prcSoFar)
    portfolio_equity = 10000 * 50  # Portfolio equity
    risk_per_trade = 0.01  # Adjusted to 1% of portfolio per trade
    
    # Volatility-based position sizing
    position_size = (portfolio_equity * risk_per_trade) / atr

    # Calculate RSI
    rsi = calculate_rsi(prcSoFar)
    latest_rsi = rsi[:, -1]
    
    # Define RSI thresholds
    overbought_threshold = 70
    oversold_threshold = 30

    # Adjust positions based on RSI
    position_adjustment = np.where((latest_rsi > overbought_threshold) | (latest_rsi < oversold_threshold), 1, 0.5)

    smoothing_factor = 0.005
    
    currentPos += (((latestTradingPositions * position_size * position_adjustment) - currentPos) * smoothing_factor).astype(int)

    # Apply trailing stop-loss
    stop_loss_percentage = 0.01  # 1% stop-loss
    for i in range(nInst):
        if currentPos[i] > 0:
            stop_price = initialPrices[i] * (1 - stop_loss_percentage)
            trailing_stop = np.max(prcSoFar[i]) * (1 - stop_loss_percentage)
            if prcSoFar[i, -1] <= stop_price or prcSoFar[i, -1] <= trailing_stop:
                currentPos[i] = 0  # Exit long position
    return currentPos
