import random
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# === Recreated penalty prediction function ===
def predict_penalty(team1, team2, model, encoder, first_shooter=None):
    try:
        team1_enc = encoder.transform([team1])[0]
        team2_enc = encoder.transform([team2])[0]
    except ValueError:
        return random.choice([team1, team2])

    if first_shooter is None:
        first_shooter = team1

    try:
        shooter_enc = encoder.transform([first_shooter])[0]
    except ValueError:
        shooter_enc = team1_enc

    X = pd.DataFrame([{
        "home_team_enc": team1_enc,
        "away_team_enc": team2_enc,
        "first_shooter_enc": shooter_enc
    }])

    pred = model.predict(X)[0]
    winner = encoder.inverse_transform([pred])[0]
    return winner

# === Load models and encoders ===
model = joblib.load("xgb_model.pkl")
le_team = joblib.load("label_encoder_teams.pkl")
le_tourn = joblib.load("label_encoder_tournaments.pkl")
le_result = joblib.load("label_encoder_results.pkl")
penalty_model = joblib.load("penalty_model.pkl")
penalty_encoder = joblib.load("label_encoder_teams.pkl")

# === Load Elo ratings ===
elo = {}
with open("elo_ratings.txt", encoding="utf-8") as f:
    for line in f:
        team, rating = line.strip().split(',')
        elo[team] = float(rating)

# === Real World Cup 2022 Results ===
REAL_RESULTS = {
    "Round of 16": [
        ("Netherlands", "United States", "Netherlands"),
        ("Argentina", "Australia", "Argentina"),
        ("Japan", "Croatia", "Croatia"),
        ("Brazil", "South Korea", "Brazil"),
        ("England", "Senegal", "England"),
        ("France", "Poland", "France"),
        ("Morocco", "Spain", "Morocco"),
        ("Portugal", "Switzerland", "Portugal")
    ],
    "Quarterfinal": [
        ("Netherlands", "Argentina", "Argentina"),
        ("Croatia", "Brazil", "Croatia"),
        ("England", "France", "France"),
        ("Morocco", "Portugal", "Morocco")
    ],
    "Semifinal": [
        ("Argentina", "Croatia", "Argentina"),
        ("France", "Morocco", "France")
    ],
    "Final": [
        ("Argentina", "France", "Argentina")
    ]
}

def simulate_match(team1, team2, tournament="FIFA World Cup", year=2022, allow_upset=True):
    try:
        X = pd.DataFrame({
            'home_team_enc': [le_team.transform([team1])[0]],
            'away_team_enc': [le_team.transform([team2])[0]],
            'tournament_enc': [le_tourn.transform([tournament])[0]],
            'year': [year],
            'elo_home': [elo.get(team1, 1500)],
            'elo_away': [elo.get(team2, 1500)],
        })
    except Exception:
        return random.choice([team1, team2])

    probs = model.predict_proba(X)[0]
    labels = le_result.classes_
    prob_dict = {label: probs[i] for i, label in enumerate(labels)}

    if allow_upset:
        roll = random.random()
        cum_prob = 0
        for label, p in sorted(prob_dict.items(), key=lambda x: x[1], reverse=True):
            cum_prob += p
            if roll < cum_prob:
                pred = label
                break
    else:
        pred = max(prob_dict, key=prob_dict.get)

    if pred == "draw":
        return f"draw → {predict_penalty(team1, team2, penalty_model, penalty_encoder)}"
    return team1 if pred == "win" else team2

def evaluate_simulation_once():
    total = 0
    correct = 0

    for stage in REAL_RESULTS:
        for team1, team2, real_winner in REAL_RESULTS[stage]:
            predicted = simulate_match(team1, team2, allow_upset=False)
            predicted_winner = predicted.split("→")[-1].strip() if "→" in predicted else predicted
            if predicted_winner == real_winner:
                correct += 1
            total += 1

    print(f"\n📊 Accuracy vs real matches: {correct}/{total} ({correct / total:.1%})")
    return correct, total

def run_monte_carlo_simulations(n_simulations=1000, allow_upset=True):
    winners = defaultdict(int)

    initial_teams = [
        "Netherlands", "United States",
        "Argentina", "Australia",
        "Japan", "Croatia",
        "Brazil", "South Korea",
        "England", "Senegal",
        "France", "Poland",
        "Morocco", "Spain",
        "Portugal", "Switzerland"
    ]

    for _ in range(n_simulations):
        teams = initial_teams.copy()
        random.shuffle(teams)

        for _ in range(4):  # 4 rounds: 16 → 8 → 4 → 2 → 1
            next_round = []
            for i in range(0, len(teams), 2):
                result = simulate_match(teams[i], teams[i+1], allow_upset=allow_upset)
                winner = result.split("→")[-1].strip() if "→" in result else result
                next_round.append(winner)
            teams = next_round
            if len(teams) == 1:
                winners[teams[0]] += 1
                break

    return winners

def plot_top_5_winners(winner_dict, filename="top5_winners.png"):
    top5 = sorted(winner_dict.items(), key=lambda x: x[1], reverse=True)[:5]
    teams, counts = zip(*top5)

    plt.figure(figsize=(10, 6))
    plt.bar(teams, counts, color='skyblue')
    plt.title("Top 5 Most Frequent Winners in 1000 Simulations")
    plt.ylabel("Number of Wins")
    plt.xlabel("Team")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"📈 Saved graph to {filename}")

# === Run everything ===
if __name__ == "__main__":
    evaluate_simulation_once()
    results = run_monte_carlo_simulations(1000, allow_upset=True)
    plot_top_5_winners(results)
