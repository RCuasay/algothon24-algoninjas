import numpy as np
import pandas as pd

nInst = 50
currentPos = np.zeros(nInst)

def calculate_rsi(prcSoFar, window=20):
    delta = np.diff(prcSoFar, axis=1)
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)
    
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
    ema = prcSoFar_df.ewm(span=20, adjust=False).mean()
    tradingPositions = (prcSoFar_df - ema).apply(np.sign)
    latestTradingPositions = tradingPositions.iloc[:, -1].to_numpy()
    
    volatility = np.array(prcSoFar).std(1)
    portfolio_equity = 10000 * 50  # Portfolio equity
    risk_per_trade = 0.01  # 1% of portfolio per trade
    position_size = (portfolio_equity * risk_per_trade) / volatility

    rsi = calculate_rsi(prcSoFar)
    latest_rsi = rsi[:, -1]
    
    overbought_threshold = 70
    oversold_threshold = 30
    position_adjustment = np.where((latest_rsi > overbought_threshold) & (latestTradingPositions < 0) |
                                    (latest_rsi < oversold_threshold) & (latestTradingPositions > 0), 1, 0.5)


    smoothing_factor = 0.01  # Smoothing factor
    
    currentPos += (((latestTradingPositions * position_size * position_adjustment) - currentPos) * smoothing_factor).astype(int)

    stop_loss_percentage = 0.025  # 2.5% stop-loss
    for i in range(nInst):
        if currentPos[i] > 0:
            trailing_stop = np.max(prcSoFar[i]) * (1 - stop_loss_percentage)
            if prcSoFar[i, -1] <= trailing_stop:
                currentPos[i] = 0  # Exit long position
    return currentPos
