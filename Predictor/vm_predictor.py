import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import math
import joblib
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb

# === Load data
df = pd.read_csv("results.csv").copy()

def get_result(row):
    if row['home_score'] > row['away_score']:
        return 'win'
    elif row['home_score'] < row['away_score']:
        return 'loss'
    return 'draw'

df['result'] = df.apply(get_result, axis=1)

# === Elo rating
elo_ratings = {}

def get_elo(team): return elo_ratings.get(team, 1500)
def get_k(t): t = t.lower(); return 60 if "world cup" in t else 50 if "qual" in t else 20 if "friendly" in t else 40
def goal_weight(hs, as_): d = abs(hs - as_); return 1.0 if d <= 1 else 1.5 if d == 2 else (11 + d) / 8
def update_elo(winner, loser, draw, hs, as_, tournament):
    ra, rb = get_elo(winner), get_elo(loser)
    ea = 1 / (1 + 10**((rb - ra)/400)); eb = 1 - ea
    k = get_k(tournament); g = goal_weight(hs, as_)
    if draw:
        ra += k * g * (0.5 - ea)
        rb += k * g * (0.5 - eb)
    else:
        ra += k * g * (1 - ea)
        rb += k * g * (0 - eb)
    elo_ratings[winner], elo_ratings[loser] = ra, rb

elo_home_list, elo_away_list = [], []

# === Form features ===
team_goaldiff_history = defaultdict(list)
home_form_list, away_form_list = [], []

for _, row in df.iterrows():
    h, a, res, t, hs, as_ = row['home_team'], row['away_team'], row['result'], row['tournament'], row['home_score'], row['away_score']

    # Elo
    elo_home_list.append(get_elo(h))
    elo_away_list.append(get_elo(a))

    # Form: average goal diff last 5
    home_form = np.mean(team_goaldiff_history[h][-5:]) if team_goaldiff_history[h] else 0
    away_form = np.mean(team_goaldiff_history[a][-5:]) if team_goaldiff_history[a] else 0
    home_form_list.append(home_form)
    away_form_list.append(away_form)

    team_goaldiff_history[h].append(hs - as_)
    team_goaldiff_history[a].append(as_ - hs)

    # Elo update
    if res == 'win': update_elo(h, a, False, hs, as_, t)
    elif res == 'loss': update_elo(a, h, False, as_, hs, t)
    else: update_elo(h, a, True, hs, as_, t)

df['elo_home'] = elo_home_list
df['elo_away'] = elo_away_list
df['elo_diff'] = df['elo_home'] - df['elo_away']  # 💡 NY feature
df['home_team_form'] = home_form_list
df['away_team_form'] = away_form_list

# === Feature Engineering
df['year'] = pd.to_datetime(df['date']).dt.year
le_team = LabelEncoder()
le_team.fit(pd.concat([df['home_team'], df['away_team']]).unique())
df['home_team_enc'] = le_team.transform(df['home_team'])
df['away_team_enc'] = le_team.transform(df['away_team'])
le_tourn = LabelEncoder()
df['tournament_enc'] = le_tourn.fit_transform(df['tournament'])
le_result = LabelEncoder()
df['result_enc'] = le_result.fit_transform(df['result'])

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

# === Compute class weights
classes = np.unique(y_train)
weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weights = dict(zip(classes, weights))
sample_weights = y_train.map(class_weights)

# === Grid Search
params = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "learning_rate": [0.01, 0.1]
}

grid = GridSearchCV(
    estimator=XGBClassifier(eval_metric='mlogloss', use_label_encoder=False),
    param_grid=params,
    cv=5,
    verbose=1,
    n_jobs=-1
)
grid.fit(X_train, y_train, sample_weight=sample_weights)
print(f"\n✅ Best parameters: {grid.best_params_}")
model = grid.best_estimator_

# === Evaluation
y_pred = model.predict(X_test)
print("\n✅ Model trained with form + elo_diff + class-weighting\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le_result.classes_, zero_division=0))
print("\n🧪 Predicted classes:", np.unique(y_pred, return_counts=True))
print("📊 Actual classes:", np.unique(y_test, return_counts=True))

# === Confusion Matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=le_result.classes_, cmap="Blues")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# === Save model and encoders
joblib.dump(model, "xgb_model.pkl")
joblib.dump(le_team, "label_encoder_teams.pkl")
joblib.dump(le_tourn, "label_encoder_tournaments.pkl")
joblib.dump(le_result, "label_encoder_results.pkl")
with open("elo_ratings.txt", "w", encoding="utf-8") as f:
    for team, rating in sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True):
        f.write(f"{team},{rating:.2f}\n")

# === Feature Importance
xgb.plot_importance(model)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.show()
