import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib
from collections import defaultdict

# === Load and preprocess data ===
df = pd.read_csv("results.csv")
def get_result(row):
    if row['home_score'] > row['away_score']: return 'win'
    elif row['home_score'] < row['away_score']: return 'loss'
    return 'draw'
df['result'] = df.apply(get_result, axis=1)

# Elo rating
elo_ratings = {}
def get_elo(team): return elo_ratings.get(team, 1500)
def get_k(t): t = t.lower(); return 60 if "world cup" in t else 50 if "qual" in t else 20 if "friendly" in t else 40
def goal_weight(hs, as_): d = abs(hs - as_); return 1.0 if d <= 1 else 1.5 if d == 2 else (11 + d) / 8
def update_elo(winner, loser, draw, hs, as_, tournament):
    ra, rb = get_elo(winner), get_elo(loser)
    ea = 1 / (1 + 10**((rb - ra)/400))
    k = get_k(tournament); g = goal_weight(hs, as_)
    if draw:
        ra += k * g * (0.5 - ea)
        rb += k * g * (0.5 - (1 - ea))
    else:
        ra += k * g * (1 - ea)
        rb += k * g * (0 - (1 - ea))
    elo_ratings[winner], elo_ratings[loser] = ra, rb

elo_home_list, elo_away_list = [], []
team_goaldiff_history = defaultdict(list)
home_form_list, away_form_list = [], []

for _, row in df.iterrows():
    h, a, res, t, hs, as_ = row['home_team'], row['away_team'], row['result'], row['tournament'], row['home_score'], row['away_score']
    elo_home_list.append(get_elo(h))
    elo_away_list.append(get_elo(a))
    home_form = np.mean(team_goaldiff_history[h][-5:]) if team_goaldiff_history[h] else 0
    away_form = np.mean(team_goaldiff_history[a][-5:]) if team_goaldiff_history[a] else 0
    home_form_list.append(home_form)
    away_form_list.append(away_form)
    team_goaldiff_history[h].append(hs - as_)
    team_goaldiff_history[a].append(as_ - hs)
    if res == 'win': update_elo(h, a, False, hs, as_, t)
    elif res == 'loss': update_elo(a, h, False, as_, hs, t)
    else: update_elo(h, a, True, hs, as_, t)

df['elo_home'] = elo_home_list
df['elo_away'] = elo_away_list
df['elo_diff'] = df['elo_home'] - df['elo_away']
df['home_team_form'] = home_form_list
df['away_team_form'] = away_form_list
df['year'] = pd.to_datetime(df['date']).dt.year

# === Encoders ===
le_team = joblib.load("label_encoder_teams.pkl")
le_tourn = joblib.load("label_encoder_tournaments.pkl")
le_result = joblib.load("label_encoder_results.pkl")
df['home_team_enc'] = le_team.transform(df['home_team'])
df['away_team_enc'] = le_team.transform(df['away_team'])
df['tournament_enc'] = le_tourn.transform(df['tournament'])
df['result_enc'] = le_result.transform(df['result'])

# === Features & split ===
features = [
    'home_team_enc', 'away_team_enc', 'tournament_enc', 'year',
    'elo_home', 'elo_away', 'elo_diff',
    'home_team_form', 'away_team_form'
]
X = df[features]
y = df['result_enc']
X_train = X[df['year'] < 2018]
y_train = y[df['year'] < 2018]
X_test = X[df['year'] >= 2018]
y_test = y[df['year'] >= 2018]

# === Skala data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Bygg ensemblemodell
xgb = XGBClassifier(eval_metric='mlogloss', learning_rate=0.01, max_depth=3, n_estimators=100)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
lr = LogisticRegression(max_iter=5000)

stack_model = StackingClassifier(
    estimators=[('xgb', xgb), ('rf', rf), ('lr', lr)],
    final_estimator=LogisticRegression(max_iter=5000),
    cv=5
)

stack_model.fit(X_train_scaled, y_train)
probs = stack_model.predict_proba(X_test_scaled)
preds = stack_model.predict(X_test_scaled)

# === Utvärdering
print("📊 Accuracy:", accuracy_score(y_test, preds))
print("📉 Log-loss:", log_loss(y_test, probs))
print("\nClassification Report:\n", classification_report(y_test, preds, target_names=le_result.classes_))

# === Exportera predictioner till CSV
decoded_preds = le_result.inverse_transform(preds)
prob_df = pd.DataFrame(probs, columns=["P_draw", "P_loss", "P_win"])
output_df = df[df['year'] >= 2018].reset_index(drop=True)[['home_team', 'away_team', 'date']]
output_df['prediction'] = decoded_preds
final_df = pd.concat([output_df, prob_df], axis=1)
final_df.to_csv("predictions.csv", index=False)
print("✅ predictions.csv sparad med sannolikheter och utfall.")
