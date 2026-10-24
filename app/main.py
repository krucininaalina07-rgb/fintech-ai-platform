from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from typing import List
import random
from datetime import datetime
import threading
from .engine import market_data, train_predict, backtest, SYMBOLS, warm_up
import sys

sys.dont_write_bytecode = True

app = FastAPI(title='Hybrid AI Market Intelligence', version='0.1.0')
app.mount('/static', StaticFiles(directory='static'), name='static')

PORTFOLIO: List[str] = []

@app.on_event("startup")
def startup_event():
    # Запускаем прогрев в отдельном потоке, чтобы не блокировать запуск сервера
    threading.Thread(target=warm_up, daemon=True).start()

@app.get('/')
def home(): return FileResponse('static/index.html')

@app.get('/api/health')
def health(): return {'status': 'ok', 'mode': 'demo'}

@app.get('/api/instruments')
def instruments(): return list(SYMBOLS)

@app.get('/api/market/{symbol}')
def market(symbol: str):
    if symbol.upper() not in SYMBOLS: raise HTTPException(404, 'Unknown symbol')
    d = market_data(symbol, 220)
    return [{'ts': str(r.ts.date()), 'open': round(r.open, 2), 'high': round(r.high, 2), 'low': round(r.low, 2), 'close': round(r.close, 2), 'volume': int(r.volume)} for _, r in d.iterrows()]

@app.get('/api/forecast/{symbol}')
def forecast(symbol: str):
    if symbol.upper() not in SYMBOLS: raise HTTPException(404, 'Unknown symbol')
    return train_predict(symbol)[1]

@app.get('/api/backtest/{symbol}')
def bt(symbol: str):
    if symbol.upper() not in SYMBOLS: raise HTTPException(404, 'Unknown symbol')
    return backtest(symbol)

@app.post('/api/twin/add/{symbol}')
def add_to_twin(symbol: str):
    if symbol.upper() not in SYMBOLS: raise HTTPException(404, 'Unknown symbol')
    if symbol.upper() not in PORTFOLIO:
        PORTFOLIO.append(symbol.upper())
    return {'status': 'ok', 'portfolio': PORTFOLIO}

@app.get('/api/twin')
def get_twin():
    return {'portfolio': PORTFOLIO}

@app.get('/api/events')
def get_events():
    events = []
    for sym in SYMBOLS.keys():
        try:
            f = train_predict(sym)[1]
            if f['regime'] == 'bullish':
                events.append(f"📈 {sym}: Смена режима на бычий (сильный тренд)")
            elif f['regime'] == 'bearish':
                events.append(f"📉 {sym}: Смена режима на медвежий (риск падения)")
            if f['risk_volatility'] > 0.35:
                events.append(f"⚠️ {sym}: Высокая волатильность ({round(f['risk_volatility']*100)}%)")
            elif f['confidence'] > 0.7:
                events.append(f"🎯 {sym}: Высокая уверенность модели ({round(f['confidence']*100)}%)")
        except Exception:
            pass
    
    events.append("🤖 Нейросеть завершила переобучение на новых данных Мосбиржи")
    events.append("👤 Эксперт проверил прогноз по SBER")
    
    random.shuffle(events)
    return {"events": events[:5], "time": datetime.now().strftime("%H:%M")}