# train_penalty_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# === Step 1: Load penalty shootout data ===
df = pd.read_csv("shootouts.csv")

# Fill missing 'first_shooter' values with 'unknown' (could be improved with logic)
df['first_shooter'] = df['first_shooter'].fillna("unknown")

# === Step 2: Encode team names ===
le_teams = LabelEncoder()
all_teams = pd.concat([df["home_team"], df["away_team"], df["first_shooter"]]).unique()
le_teams.fit(all_teams)

df["home_team_enc"] = le_teams.transform(df["home_team"])
df["away_team_enc"] = le_teams.transform(df["away_team"])
df["first_shooter_enc"] = le_teams.transform(df["first_shooter"])
df["winner_enc"] = le_teams.transform(df["winner"])

# === Step 3: Train Random Forest classifier ===
X = df[["home_team_enc", "away_team_enc", "first_shooter_enc"]]
y = df["winner_enc"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# === Step 4: Save model and encoder ===
joblib.dump(model, "penalty_model.pkl")
joblib.dump(le_teams, "label_encoder_teams.pkl")

print("✅ Penalty shootout AI model trained and saved!")

# === Step 5: Define prediction function ===
def predict_penalty(team1, team2, model, encoder, first_shooter=None):
    """
    Predicts the winner of a penalty shootout between two teams.
    
    Parameters:
    - team1: Home team (string)
    - team2: Away team (string)
    - model: Trained classifier
    - encoder: LabelEncoder for teams
    - first_shooter: Optional. Specify which team takes first penalty
    
    Returns:
    - Predicted winning team (string)
    """
    try:
        team1_enc = encoder.transform([team1])[0]
        team2_enc = encoder.transform([team2])[0]
    except ValueError as e:
        raise ValueError("One of the teams is unknown to the encoder.") from e

    if first_shooter is None:
        # Default: assume home team goes first
        first_shooter = team1

    try:
        first_shooter_enc = encoder.transform([first_shooter])[0]
    except ValueError:
        first_shooter_enc = -1  # Unknown team (should be avoided)

    features = pd.DataFrame([{
        "home_team_enc": team1_enc,
        "away_team_enc": team2_enc,
        "first_shooter_enc": first_shooter_enc
    }])

    pred_enc = model.predict(features)[0]
    pred_team = encoder.inverse_transform([pred_enc])[0]
    return pred_team
