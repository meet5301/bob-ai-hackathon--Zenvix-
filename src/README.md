# Grid Guard — `src/` Folder

Power Outage Prediction & Grid Equipment Failure Advisor  
**Team Maverick** — IBM Bobathon U1

---

## Subfolder Layout

| Folder | Purpose |
|---|---|
| `data_generation/` | Produces synthetic asset, sensor, and incident CSV data; fetches real historical weather from Open-Meteo |
| `etl/` | Postgres schema definition; extract → transform → load pipeline that pushes all data into Neon |
| `feature_engineering/` | Reads cleaned data from Neon, computes rolling sensor statistics, and builds the final model training table |
| `ml/` | Trains an XGBoost failure-prediction classifier; runs inference and ranks assets by severity score |
| `llm/` | Calls IBM watsonx.ai (Bob) to generate a prioritised maintenance and crew pre-positioning plan from ranked assets |
| `api/` | FastAPI service exposing `/health`, `/predictions`, `/rankings`, and `/plan` endpoints consumed by the dashboard |
| `dashboard/` | Streamlit app with an asset risk map, ranked risk table, and the watsonx-generated maintenance plan |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and fill in environment variables
cp .env.example .env
# edit .env with your Neon connection string, watsonx credentials, etc.

# 3. Generate synthetic data and fetch weather
python -m data_generation.generate_fake_data
python -m data_generation.fetch_weather

# 4. Run the ETL pipeline (creates schema, loads all tables into Neon)
python -m etl.load_to_neon

# 5. Build features and train the model
python -m feature_engineering.build_features
python -m ml.train_model

# 6. Start the API
uvicorn api.main:app --reload

# 7. Start the dashboard (in a second terminal)
streamlit run dashboard/app.py
```
