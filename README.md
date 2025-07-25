# Football Match Predictor

This is a simple simulator for predicting international football tournaments using machine learning. It uses a trained XGBoost model together with Elo ratings and recent team form to simulate a full World Cup – including group stage and knockout rounds.

## What it does

- Lets you select 32 national teams
- Automatically assigns teams to groups A–H
- Simulates group matches using a trained model
- Calculates group standings based on simulated results
- Runs knockout rounds based on group rankings
- Uses a separate model to predict penalty shootouts
- Displays detailed results and most frequent winners

## How to run

1. Make sure you have Python 3.9 or later.
2. Go to the `Predictor` folder:

cd path/to/Predictor

    Install required packages:

pip install pandas numpy scikit-learn xgboost joblib

    Start the GUI:

python main_gui.py

How it works

    The AI model predicts outcomes based on:

        Elo rating for each team

        Elo difference between teams

        Recent goal difference (last 5 games)

        Encoded tournament and team info

    Elo ratings are updated after each match

    Group standings are calculated using points, goal difference, and goals for

    Knockout matches use the same model

    If a match is drawn, a penalty shootout model predicts the winner

## Files

| File                        | Description                             |
|-----------------------------|-----------------------------------------|
| `main_gui.py`               | GUI app for simulations                 |
| `simulate_knockout_rounds.py` | Script for batch simulation runs     |
| `results.csv`               | Historical match data                  |
| `xgb_model.pkl`             | Trained XGBoost model                  |
| `penalty_model.pkl`         | Penalty shootout model                 |
| `label_encoder_*.pkl`       | Encoders for teams, tournaments, outcomes |

    Based on men’s full international matches only

    Does not include Olympic, youth, or B-team matches

    Built for learning, experimentation and fun – not for betting
