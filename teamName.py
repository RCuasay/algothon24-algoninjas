import numpy as np
from sklearn.ensemble import RandomForestClassifier

nInst = 50
currentPos = np.zeros(nInst)
model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
model_trained = False

def ema(data, span):
    alpha = 2 / (span + 1)
    alpha_rev = 1 - alpha
    n = data.shape[1]
    pows = alpha_rev**(np.arange(n))
    scale_arr = 1/pows
    offset = data[:, 0][:, np.newaxis] * alpha_rev**(np.arange(n))
    pw0 = alpha*alpha_rev**(np.arange(n-1, -1, -1))
    mult = data * pw0
    cumsums = mult.cumsum(axis=1)
    out = offset + cumsums*scale_arr
    return out

def getMyPosition(prcSoFar):
    global currentPos, model, model_trained
    (nins, nt) = prcSoFar.shape

    # Dynamic lookback period based on market volatility
    vol = np.std(np.log(prcSoFar[:, -50:] / prcSoFar[:, -51:-1]), axis=1)
    lookback = np.clip(int(100 / np.median(vol)), 50, 200)

    if nt < lookback + 1:
        return np.zeros(nins)

    returns = np.log(prcSoFar[:, 1:] / prcSoFar[:, :-1])

    short_ema = ema(prcSoFar, span=20)
    long_ema = ema(prcSoFar, span=50)

    z_score = (prcSoFar[:, -1] - long_ema[:, -1]) / np.std(prcSoFar[:, -lookback:], axis=1)
    trend = (short_ema[:, -1] - long_ema[:, -1]) / long_ema[:, -1]
    momentum = np.sum(returns[:, -20:], axis=1)

    # Volatility breakout
    vol_lookback = 20
    historical_vol = np.std(returns[:, -vol_lookback:], axis=1)
    current_vol = np.abs(returns[:, -1])
    vol_breakout = (current_vol - historical_vol) / historical_vol

    # Sector rotation
    sector_size = 10
    sector_returns = np.array([np.mean(returns[i:i+sector_size, -1]) for i in range(0, nins, sector_size)])
    sector_signal = np.repeat(sector_returns, sector_size)

    # Pair trading
    correlations = np.corrcoef(returns)
    pair_signal = np.zeros(nins)
    for i in range(nins):
        most_correlated = np.argsort(correlations[i])[-2]  # Second highest correlation (highest is self)
        pair_spread = prcSoFar[i, -1] / prcSoFar[most_correlated, -1] - np.mean(prcSoFar[i, -lookback:] / prcSoFar[most_correlated, -lookback:])
        pair_signal[i] = -np.sign(pair_spread)  # Go against the spread

    features = np.column_stack([z_score, trend, momentum, vol_breakout, sector_signal, pair_signal])

    if not model_trained and nt > lookback + 20:
        X = features
        y = np.sign(returns[:, -1])
        model.fit(X, y)
        model_trained = True

    if model_trained:
        signal = model.predict_proba(features)[:, 1] * 2 - 1
    else:
        signal = -0.2*z_score + 0.3*trend + 0.2*momentum + 0.1*vol_breakout + 0.1*sector_signal + 0.1*pair_signal

    # Risk parity position sizing
    risk_contrib = 1 / (np.std(returns[:, -lookback:], axis=1) + 1e-6)
    risk_contrib /= np.sum(risk_contrib)

    target_vol = 0.025  # Slightly increased target volatility
    realized_vol = np.std(returns[:, -lookback:], axis=1) * np.sqrt(252)
    vol_scalar = target_vol / realized_vol

    signal *= vol_scalar * risk_contrib

    max_change = 1400  # Slightly increased maximum position change
    position_change = np.clip(signal, -1, 1) * max_change
    position_change = np.array([int(x) for x in position_change / prcSoFar[:, -1]])

    max_position = 6000  # Slightly increased maximum position size
    new_pos = np.clip(currentPos + position_change, -max_position, max_position)

    stop_loss_threshold = 0.045  # Slightly relaxed stop loss
    trailing_stop_threshold = 0.035  # Slightly relaxed trailing stop

    for i in range(nins):
        if currentPos[i] != 0:
            entry_price = prcSoFar[i, -lookback]
            current_price = prcSoFar[i, -1]
            price_change = (current_price - entry_price) / entry_price

            if (currentPos[i] > 0 and price_change < -stop_loss_threshold) or \
               (currentPos[i] < 0 and price_change > stop_loss_threshold) or \
               (currentPos[i] > 0 and price_change > trailing_stop_threshold and
                (current_price / np.max(prcSoFar[i, -lookback:]) - 1) < -trailing_stop_threshold) or \
               (currentPos[i] < 0 and price_change < -trailing_stop_threshold and
                (current_price / np.min(prcSoFar[i, -lookback:]) - 1) > trailing_stop_threshold):
                new_pos[i] = 0

    currentPos = new_pos
    return currentPos
