# World Cup Match Predictor (XGBoost + Elo)

This project simulates an entire international football tournament using machine learning. It combines a trained XGBoost classifier with Elo ratings and recent team form to predict match outcomes. The simulator includes both group stages and knockout rounds, just like a real FIFA World Cup.

## Features

- Select 32 national teams and assign them to groups A–H
- Predict match outcomes using a trained AI model
- Use Elo ratings and recent goal difference as features
- Simulate group stages and knockout brackets
- Predict penalty shootouts using a separate model
- Display group tables, match summaries, and final winners

## How to run


pip install pandas numpy scikit-learn xgboost joblib
python main_gui.py

## Files

| File                        | Description                             |
|-----------------------------|-----------------------------------------|
| `main_gui.py`               | GUI app for simulations                 |
| `simulate_knockout_rounds.py` | Script for batch simulation runs     |
| `results.csv`               | Historical match data                  |
| `xgb_model.pkl`             | Trained XGBoost model                  |
| `penalty_model.pkl`         | Penalty shootout model                 |
| `label_encoder_*.pkl`       | Encoders for teams, tournaments, outcomes |


Includes over 47,000 men's full international matches from 1872 to 2024.
Excludes Olympic Games, youth teams, and B-teams.

# Notes

    Built for experimentation and fun

    Not intended for betting or gambling

