import pandas as pd
import joblib
import random
from texttable import Texttable

# === Load the trained penalty shootout model and team encoder ===
penalty_model = joblib.load("penalty_model.pkl")
penalty_encoder = joblib.load("label_encoder_teams.pkl")

# === Predict penalty winner between two teams ===
def predict_penalty_winner(team1, team2):
    """
    Predicts the winner of a penalty shootout between two teams.
    If prediction fails (e.g., team not in encoder), returns a random winner.
    """
    try:
        team1_enc = penalty_encoder.transform([team1])[0]
        team2_enc = penalty_encoder.transform([team2])[0]
        first_shooter_enc = team1_enc  # Assume team1 shoots first

        # Create feature vector for prediction
        X_penalty = [[team1_enc, team2_enc, first_shooter_enc]]

        winner_enc = penalty_model.predict(X_penalty)[0]
        winner = penalty_encoder.inverse_transform([winner_enc])[0]
        return winner
    except Exception as e:
        print(f"⚠️ Penalty prediction error: {e}")
        return random.choice([team1, team2])

# === Simulate a single match (90 mins + penalties if draw) ===
def simulate_match(team1, team2):
    """
    Simulates a match between team1 and team2.
    Returns:
        result_type: 'win', 'loss', or 'draw'
        winner: name of winning team or draw with penalty outcome
    """
    outcomes = ["win", "loss", "draw"]
    weights = [0.4, 0.4, 0.2]  # Adjust for realism
    result = random.choices(outcomes, weights=weights)[0]

    if result == "draw":
        penalty_winner = predict_penalty_winner(team1, team2)
        return result, f"draw → {penalty_winner}"
    elif result == "win":
        return result, team1
    else:
        return result, team2

# === Knockout tournament: Quarterfinals → Semifinals → Final ===
def knockout_stage(teams):
    """
    Runs a single-elimination tournament with the given teams.
    """
    stage_names = ["QUARTERFINALS", "SEMIFINALS", "FINAL"]
    stage = 0

    while len(teams) > 1:
        print(f"\n=== {stage_names[stage]} ===")
        table = Texttable()
        table.header(["Team 1", "", "Team 2", "", "Result"])
        winners = []

        for i in range(0, len(teams), 2):
            team1, team2 = teams[i], teams[i + 1]
            _, result = simulate_match(team1, team2)
            table.add_row([team1, "vs", team2, "→", result])

            if "→" in result:
                winners.append(result.split("→")[-1].strip())
            else:
                winners.append(result)

        print(table.draw())
        teams = winners
        stage += 1

    print(f"\n🏆 Tournament Winner: {teams[0]} 🏆")

# === Run simulation with example teams ===
def run():
    teams = [
        "Brazil", "France", "Germany", "Argentina",
        "England", "Spain", "Japan", "Sweden"
    ]
    random.shuffle(teams)
    knockout_stage(teams)

if __name__ == "__main__":
    run()
