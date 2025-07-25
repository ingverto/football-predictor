import pandas as pd
import numpy as np
import joblib
import random
from collections import Counter, defaultdict
from sklearn.preprocessing import StandardScaler

# === Load models and encoders ===
model = joblib.load("xgb_model.pkl")
le_team = joblib.load("label_encoder_teams.pkl")
le_tourn = joblib.load("label_encoder_tournaments.pkl")
le_result = joblib.load("label_encoder_results.pkl")
penalty_model = joblib.load("penalty_model.pkl")
penalty_encoder = joblib.load("label_encoder_teams.pkl")  # Used for penalty model
group_tables = joblib.load("group_tables.pkl")

# === Load historical results for Elo and form calculation ===
results_df = pd.read_csv("results.csv")

def get_result(row):
    if row['home_score'] > row['away_score']:
        return 'win'
    elif row['home_score'] < row['away_score']:
        return 'loss'
    return 'draw'
results_df['result'] = results_df.apply(get_result, axis=1)

elo_ratings = {}
team_goaldiff_history = defaultdict(list)

def get_elo(team):
    return elo_ratings.get(team, 1500)

def get_k(tournament):
    t = tournament.lower()
    if "world cup" in t:
        return 60
    elif "qual" in t:
        return 50
    elif "friendly" in t:
        return 20
    return 40

def goal_weight(hs, as_):
    d = abs(hs - as_)
    if d <= 1:
        return 1.0
    elif d == 2:
        return 1.5
    return (11 + d) / 8

def update_elo(winner, loser, draw, hs, as_, tournament):
    ra, rb = get_elo(winner), get_elo(loser)
    ea = 1 / (1 + 10**((rb - ra)/400))
    k = get_k(tournament)
    g = goal_weight(hs, as_)
    if draw:
        ra += k * g * (0.5 - ea)
        rb += k * g * (0.5 - (1 - ea))
    else:
        ra += k * g * (1 - ea)
        rb += k * g * (0 - (1 - ea))
    elo_ratings[winner], elo_ratings[loser] = ra, rb

# Update Elo ratings and goal difference history from results
for _, row in results_df.iterrows():
    h, a, res, t, hs, as_ = row['home_team'], row['away_team'], row['result'], row['tournament'], row['home_score'], row['away_score']
    update_elo(h, a, res == 'draw', hs, as_, t)
    team_goaldiff_history[h].append(hs - as_)
    team_goaldiff_history[a].append(as_ - hs)

scaler = StandardScaler()

# === Predict penalty shootout winner using AI model ===
def predict_penalty(team1, team2, model, encoder, first_shooter=None):
    """
    Predicts the winner of a penalty shootout between two teams using an AI model.
    Falls back to Elo if teams are unknown or model output is invalid.
    """
    if team1 not in encoder.classes_ or team2 not in encoder.classes_:
        return team1 if get_elo(team1) >= get_elo(team2) else team2

    if first_shooter is None:
        first_shooter = team1

    try:
        team1_enc = encoder.transform([team1])[0]
        team2_enc = encoder.transform([team2])[0]
        first_enc = encoder.transform([first_shooter])[0]
    except Exception:
        return team1 if get_elo(team1) >= get_elo(team2) else team2

    features = pd.DataFrame([{
        "home_team_enc": team1_enc,
        "away_team_enc": team2_enc,
        "first_shooter_enc": first_enc
    }])

    try:
        pred_enc = model.predict(features)[0]
        pred_team = encoder.inverse_transform([pred_enc])[0]
    except Exception:
        return team1 if get_elo(team1) >= get_elo(team2) else team2

    # Ensure the model returns a valid team
    if pred_team not in [team1, team2]:
        print(f"⚠️ AI penalty model returned invalid team: {pred_team} — fallback to Elo")
        return team1 if get_elo(team1) >= get_elo(team2) else team2

    return pred_team

# === Simulate a single knockout match ===
def simulate_knockout(team1, team2):
    if team1 not in le_team.classes_ or team2 not in le_team.classes_:
        return random.choice([team1, team2])

    X = pd.DataFrame([{
        "home_team_enc": le_team.transform([team1])[0],
        "away_team_enc": le_team.transform([team2])[0],
        "tournament_enc": le_tourn.transform(["FIFA World Cup"])[0],
        "year": 2022,
        "elo_home": get_elo(team1),
        "elo_away": get_elo(team2),
        "elo_diff": get_elo(team1) - get_elo(team2),
        "home_team_form": np.mean(team_goaldiff_history[team1][-5:]) if team_goaldiff_history[team1] else 0,
        "away_team_form": np.mean(team_goaldiff_history[team2][-5:]) if team_goaldiff_history[team2] else 0
    }])
    X_scaled = scaler.fit_transform(X)
    probs = model.predict_proba(X_scaled)[0]
    outcome = np.random.choice(le_result.classes_, p=probs)

    if outcome == "win":
        return team1
    elif outcome == "loss":
        return team2
    else:
        # Use AI penalty model for shootout
        return predict_penalty(team1, team2, penalty_model, penalty_encoder)

# === Rank group stage results and set up knockout rounds ===
group_winners = {}
group_runnersup = {}
for group, table in group_tables.items():
    sorted_teams = table.sort_values("Pts", ascending=False).index.tolist()
    if len(sorted_teams) >= 2:
        group_winners[group] = sorted_teams[0]
        group_runnersup[group] = sorted_teams[1]

# === Define Round of 16 pairings ===
round_of_16 = [
    (group_winners['A'], group_runnersup['B']),
    (group_winners['C'], group_runnersup['D']),
    (group_winners['E'], group_runnersup['F']),
    (group_winners['G'], group_runnersup['H']),
    (group_winners['B'], group_runnersup['A']),
    (group_winners['D'], group_runnersup['C']),
    (group_winners['F'], group_runnersup['E']),
    (group_winners['H'], group_runnersup['G']),
]

# === Simulate 1000 World Cups ===
final_winners = []
for _ in range(1000):
    # Quarterfinals
    quarter_finalists = [simulate_knockout(a, b) for a, b in round_of_16]
    qf_pairs = [(quarter_finalists[i], quarter_finalists[i+1]) for i in range(0, 8, 2)]
    # Semifinals
    semi_finalists = [simulate_knockout(a, b) for a, b in qf_pairs]
    sf_pairs = [(semi_finalists[0], semi_finalists[1]), (semi_finalists[2], semi_finalists[3])]
    # Finalists
    finalists = [simulate_knockout(a, b) for a, b in sf_pairs]
    # Final
    winner = simulate_knockout(finalists[0], finalists[1])
    final_winners.append(winner)

# === Print results ===
print("\n🏆 Win frequency in 1000 World Cups (with AI penalty model):\n")
for team, count in Counter(final_winners).most_common():
    print(f"{team}: {count} wins")
