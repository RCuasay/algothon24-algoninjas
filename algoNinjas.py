import numpy as np
import pandas as pd

nInst = 50
currentPos = np.zeros(nInst)

def calculate_atr(prcSoFar, window=20):
    high = prcSoFar.max(axis=1)
    low = prcSoFar.min(axis=1)
    close = prcSoFar[:, -1]
    
    tr = np.maximum(high - low, np.maximum(abs(high - close), abs(low - close)))
    atr = pd.Series(tr).rolling(window=window, min_periods=1).mean().values
    return atr

def getMyPosition(prcSoFar):
    global currentPos
    (nins, nt) = prcSoFar.shape
    if nt < 2:
        return np.zeros(nins)
    
    prcSoFar_df = pd.DataFrame(prcSoFar)
    ema = prcSoFar_df.ewm(span=20, adjust=False).mean()
    tradingPositions = (prcSoFar_df - ema).apply(np.sign)
    latestTradingPositions = tradingPositions.iloc[:, -1].to_numpy()
    
    # Calculate ATR for volatility-based position sizing
    atr = calculate_atr(prcSoFar)
    portfolio_equity = 10000 * 50  # Portfolio equity
    risk_per_trade = 0.01  # Adjusted to 1% of portfolio per trade
    
    # Volatility-based position sizing
    position_size = (portfolio_equity * risk_per_trade) / atr
    
    currentPos += (((latestTradingPositions * position_size) - currentPos) * 0.01).astype(int)
    
    return currentPos
