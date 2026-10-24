from fastapi.testclient import TestClient
from app.main import app
c=TestClient(app)
def test_health(): assert c.get('/api/health').json()['status']=='ok'
def test_forecast():
    x=c.get('/api/forecast/AAPL'); assert x.status_code==200; j=x.json(); assert 0<=j['p_up']<=1 and len(j['factors'])==5
def test_market(): assert len(c.get('/api/market/NVDA').json())==220
def test_unknown(): assert c.get('/api/forecast/XXX').status_code==404
