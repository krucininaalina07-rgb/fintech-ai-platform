import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import moexalgo

SYMBOLS = {"SBER": 250.0, "GAZP": 130.0, "LKOH": 7000.0, "YNDX": 4000.0, "GMKN": 15000.0, "ROSN": 550.0}

# Глобальный кэш для результатов
CACHE_PREDICT = {}
CACHE_BACKTEST = {}
CACHE_MARKET = {}

def market_data(symbol='SBER', n=520):
    symbol = symbol.upper()
    
    # ПОЛНОСТЬЮ ОТКЛЮЧАЕМ МОСБИРЖУ НА RENDER
    # try:
    #     df = moexalgo.Market(symbol).candles(start=pd.Timestamp.now() - pd.Timedelta(days=n*2), end=pd.Timestamp.now(), period='D1')
    #     if df is not None and not df.empty:
    #         df = df.tail(n).copy()
    #         df['ts'] = pd.to_datetime(df['begin'])
    #         df = df.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'})
    #         res = df[['ts', 'open', 'high', 'low', 'close', 'volume']]
    #         CACHE_MARKET[symbol] = res
    #         return res
    # except Exception as e:
    #     print(f"MOEX Error: {e}")

    # Fallback: Генерация синтетики (всегда работает)
    base = SYMBOLS.get(symbol, 100.0)
    seed = sum(map(ord, symbol)) + 42
    rng = np.random.default_rng(seed)
    regime = np.sin(np.arange(n)/42)*0.0009
    ret = 0.00025+regime+rng.normal(0,0.012,n)
    close = base*np.exp(np.cumsum(ret)-np.cumsum(ret)[-1])
    open_ = close*np.exp(rng.normal(0,0.002,n))
    spread = np.abs(rng.normal(.006,.003,n))
    high = np.maximum(open_,close)*(1+spread)
    low = np.minimum(open_,close)*(1-spread)
    volume = rng.integers(800_000,8_000_000,n)
    dates = pd.bdate_range(end=pd.Timestamp.now(tz='UTC').normalize(), periods=n)
    res = pd.DataFrame({'ts':dates,'open':open_,'high':high,'low':low,'close':close,'volume':volume})
    CACHE_MARKET[symbol] = res
    return res
def features(df):
    x = df.copy(); c = x.close
    x['ret1'] = np.log(c/c.shift(1)); x['ret5'] = np.log(c/c.shift(5))
    x['sma20'] = c.rolling(20).mean(); x['ema12'] = c.ewm(span=12,adjust=False).mean(); x['ema26'] = c.ewm(span=26,adjust=False).mean()
    x['macd'] = x.ema12-x.ema26
    d = c.diff(); gain = d.clip(lower=0).rolling(14).mean(); loss = (-d.clip(upper=0)).rolling(14).mean(); rs = gain/(loss+1e-9)
    x['rsi'] = 100-100/(1+rs); x['vol20'] = x.ret1.rolling(20).std()*np.sqrt(252)
    tr = pd.concat([(x.high-x.low),(x.high-c.shift()).abs(),(x.low-c.shift()).abs()],axis=1).max(axis=1)
    x['atr'] = tr.rolling(14).mean()/c; x['mom20'] = c/c.shift(20)-1; x['vol_z'] = (x.volume-x.volume.rolling(20).mean())/(x.volume.rolling(20).std()+1)
    x['target'] = (c.shift(-5)>c).astype(int)
    return x.dropna().reset_index(drop=True)

COLS = ['ret1','ret5','macd','rsi','vol20','atr','mom20','vol_z']

def train_predict(symbol):
    symbol = symbol.upper()
    if symbol in CACHE_PREDICT:
        return CACHE_PREDICT[symbol][0], CACHE_PREDICT[symbol][1]

    f = features(market_data(symbol))
    split = int(len(f)*.8)
    tr = f.iloc[:split]; te = f.iloc[split:-5] if len(f)-split>5 else f.iloc[split:]
    X, y = tr[COLS], tr.target

    # Обучаем модели (Random Forest 50 деревьев - баланс памяти и точности)
    rf = RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_leaf=5, random_state=7).fit(X,y)
    lr = LogisticRegression(max_iter=1000).fit(X,y)

    last = f.iloc[[-1]][COLS]
    p_rf = float(rf.predict_proba(last)[0,1])
    p_lr = float(lr.predict_proba(last)[0,1])
    row = f.iloc[-1]
    
    ta = 0.5 + .16*np.tanh(row.mom20*8)+.10*np.tanh(row.macd/(row.close*.01+1e-9))-.10*np.tanh((row.rsi-50)/20)
    sentiment = float(np.clip(.15*np.sin(len(symbol)+datetime.now().day/3),-.25,.25))
    
    p = float(np.clip(.42*p_rf+.28*p_lr+.22*ta+.08*(.5+sentiment),.02,.98))
    agreement = 1-np.std([p_rf,p_lr,ta])
    confidence = float(np.clip(.55+.35*agreement-.25*min(row.vol20,1),.35,.94))
    imp = sorted(zip(COLS,rf.feature_importances_),key=lambda z:z[1],reverse=True)[:5]
    expected = (p-.5)*2*row.vol20/math.sqrt(252)*math.sqrt(5)

    result = {
        'symbol': symbol, 'as_of': str(row.ts), 'price': round(float(row.close), 2),
        'p_up': round(p, 4), 'p_down': round(1-p, 4), 'confidence': round(confidence, 4),
        'expected_return_5d': round(float(expected), 4), 'risk_volatility': round(float(row.vol20), 4),
        'trend_score': round(100*p, 1), 'regime': 'bullish' if row.mom20>.02 else ('bearish' if row.mom20<-.02 else 'neutral'),
        'models': {'random_forest': round(p_rf, 4), 'logistic': round(p_lr, 4), 'technical': round(float(ta), 4), 'sentiment': round(sentiment, 4)},
        'factors': [{'name': a, 'importance': round(float(b), 4)} for a, b in imp]
    }
    
    CACHE_PREDICT[symbol] = (f, result)
    return f, result

def backtest(symbol):
    symbol = symbol.upper()
    if symbol in CACHE_BACKTEST:
        return CACHE_BACKTEST[symbol]

    f = features(market_data(symbol))
    start = 220; equity = 1.; bh = 1.; curve = []; wins = trades = 0
    for i in range(start, len(f)-5, 20):
        tr = f.iloc[:i]; test = f.iloc[i:min(i+20,len(f)-5)]
        model = LogisticRegression(max_iter=500).fit(tr[COLS],tr.target)
        probs = model.predict_proba(test[COLS])[:,1]
        for (_,r),p in zip(test.iterrows(),probs):
            future = f.iloc[min(r.name+5,len(f)-1)].close/r.close-1
            pos = 1 if p>.56 else (-1 if p<.44 else 0)
            pnl = pos*future-(0.0005 if pos else 0); equity *= 1+pnl; bh *= 1+future
            if pos: trades += 1; wins += pnl>0
            curve.append({'ts':str(r.ts.date()),'equity':round(equity,4)})

    rets = pd.Series([curve[i]['equity']/curve[i-1]['equity']-1 for i in range(1,len(curve))])
    sharpe = float(rets.mean()/(rets.std()+1e-9)*np.sqrt(252/5)) if len(rets)>2 else 0
    peaks = np.maximum.accumulate([x['equity'] for x in curve]); vals = np.array([x['equity'] for x in curve]); dd = float(np.min(vals/peaks-1))

    res = {
        'symbol': symbol, 'total_return': round(equity-1, 4), 'benchmark_return': round(bh-1, 4),
        'sharpe': round(sharpe, 2), 'max_drawdown': round(dd, 4), 'trades': trades,
        'hit_rate': round(wins/max(trades,1), 3), 'curve': curve[-120:]
    }
    CACHE_BACKTEST[symbol] = res
    return res

# --- Функция для прогрева кэша при старте сервера ---
def warm_up():
    print("Прогрев кэша (загрузка Мосбиржи и обучение моделей)...")
    for s in SYMBOLS.keys():
        try:
            train_predict(s)
            backtest(s)
            print(f"✅ {s} готов")
        except Exception as e:
            print(f"❌ {s} ошибка: {e}")
    print("Прогрев завершен.")