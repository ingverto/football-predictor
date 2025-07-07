import random
import joblib
import pandas as pd
from xgboost import XGBClassifier
from texttable import Texttable
from train_penalty_model import predict_penalty

# === Load trained models and encoders ===
model = joblib.load("xgb_model.pkl")
le_team = joblib.load("label_encoder_teams.pkl")
le_tourn = joblib.load("label_encoder_tournaments.pkl")
le_result = joblib.load("label_encoder_results.pkl")

penalty_model = joblib.load("penalty_model.pkl")
penalty_encoder = joblib.load("label_encoder_teams.pkl")  # same encoder for teams

# === Load Elo ratings from file ===
elo = {}
with open("elo_ratings.txt", encoding="utf-8") as f:
    for line in f:
        team, rating = line.strip().split(',')
        elo[team] = float(rating)

# === Simulate a single match (AI prediction + optional upset + penalties) ===
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
    except Exception as e:
        print(f"⚠️ Error encoding input data: {e}")
        return random.choice([team1, team2])

    # Predict outcome probabilities
    probs = model.predict_proba(X)[0]
    labels = le_result.classes_
    prob_dict = {label: probs[i] for i, label in enumerate(labels)}

    # Allow upset simulation (sample by probability)
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

    # Handle penalty shootout if draw
    if pred == "draw":
        winner = predict_penalty(team1, team2, penalty_model, penalty_encoder)
        return f"draw → {winner}"
    else:
        return team1 if pred == "win" else team2

# === Knockout round: one stage of the tournament ===
def knockout(teams, round_name, allow_upset=True):
    print(f"\n=== {round_name.upper()} ===")
    table = Texttable()
    table.add_row(["Team 1", "", "Team 2", "", "Result"])
    next_round = []

    for i in range(0, len(teams), 2):
        team1, team2 = teams[i], teams[i+1]
        result = simulate_match(team1, team2, allow_upset=allow_upset)

        # Extract winner
        if "→" in result:
            winner = result.split("→")[-1].strip()
        else:
            winner = result

        next_round.append(winner)
        table.add_row([team1, "vs", team2, "→", result])

    print(table.draw())
    return next_round

# === Run a predefined World Cup 2022 bracket ===
def run_vm2022_test():
    print("\n=== World Cup 2022 Knockout Simulation ===")

    # Ask user if upsets should be allowed
    user_input = input("Allow upsets? (y/n): ").strip().lower()
    allow_upset = (user_input == "y")
    print(f"\nUpsets enabled: {'✅' if allow_upset else '❌'}")

    # Round of 16 (predefined bracket)
    teams = [
        "Netherlands", "United States",
        "Argentina", "Australia",
        "Japan", "Croatia",
        "Brazil", "South Korea",
        "England", "Senegal",
        "France", "Poland",
        "Morocco", "Spain",
        "Portugal", "Switzerland"
    ]

    stages = ["Round of 16", "Quarterfinal", "Semifinal", "Final"]

    for stage in stages:
        teams = knockout(teams, stage, allow_upset=allow_upset)
        if len(teams) == 1:
            print(f"\n🏆 World Cup Winner: {teams[0]} 🏆")
            break

# === Entry point ===
if __name__ == "__main__":
    run_vm2022_test()

# Optional: Save the booster as JSON if needed
# model.get_booster().save_model("xgb_model.json")
