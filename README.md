# Hybrid AI Market Intelligence — MVP

Демонстрационный MVP гибридной ИИ-платформы биржевой аналитики. Работает без API-ключей: использует детерминированные синтетические OHLCV, рассчитывает технические признаки, объединяет Random Forest, Logistic Regression, технический сигнал и sentiment-компонент, возвращает probability/confidence/explainability и выполняет walk-forward backtest.

## Запуск
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Откройте `http://127.0.0.1:8000`. Swagger: `/docs`.

## API
- `GET /api/health`
- `GET /api/instruments`
- `GET /api/market/{symbol}`
- `GET /api/forecast/{symbol}`
- `GET /api/backtest/{symbol}`

## Важно
Это инженерный демонстратор, не инвестиционная рекомендация. Перед production: подключить point-in-time market/news feeds, PostgreSQL/TimescaleDB, аутентификацию, model registry, calibration/drift monitoring, корпоративные события, реалистичный slippage и строгую out-of-sample оценку.
