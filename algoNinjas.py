import numpy as np
import pandas as pd

##### TODO #########################################
### RENAME THIS FILE TO YOUR TEAM NAME #############
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50
currentPos = np.zeros(nInst)

def getMyPosition(prcSoFar):
    global currentPos
    (nins, nt) = prcSoFar.shape
    if nt < 2:
        return np.zeros(nins)
    
    # Convert prcSoFar to DataFrame for easier manipulation
    prcSoFar = pd.DataFrame(prcSoFar)
    
    # Calculate the Exponential Moving Average (EMA)
    ema = prcSoFar.ewm(span=50, adjust=False).mean()
    
    # Generate trading positions based on the sign of (price - EMA)
    tradingPositions = (prcSoFar - ema).apply(np.sign)
    
    # Latest trading positions
    latestTradingPositions = tradingPositions.iloc[:, -1]
    
    # Update current positions
    currentPos = latestTradingPositions.to_numpy()
    
    return currentPos
