import tkinter as tk
from tkinter import ttk
import random
import pandas as pd
import numpy as np
import joblib
from collections import defaultdict, Counter
import threading

# === Load models and encoders ===
model = joblib.load("xgb_model.pkl")
le_team = joblib.load("label_encoder_teams.pkl")
le_tourn = joblib.load("label_encoder_tournaments.pkl")
le_result = joblib.load("label_encoder_results.pkl")
penalty_model = joblib.load("penalty_model.pkl")
penalty_encoder = joblib.load("label_encoder_teams.pkl")

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

pd.set_option('future.no_silent_downcasting', True)

class VMSimulatorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("⚽ World Cup Simulator with AI")
        self.geometry("1000x700")

        # State variables
        self.selected_teams = []  # Order of selected teams
        self.group_tables = {}    # Group tables after simulation
        self.final_winners = Counter()
        self.group_sim_count = 1  # Matches per group meeting

        self.create_widgets()

    def create_widgets(self):
        # === Main layout frame
        main_frame = tk.Frame(self)
        main_frame.pack(fill="both", expand=True)

        # === Left column: team selection and group info
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side="left", fill="y", padx=10, pady=10)

        self.label = ttk.Label(left_frame, text="Select 32 teams:")
        self.label.pack()

        # Listbox with scrollbar and group labels
        listbox_frame = tk.Frame(left_frame)
        listbox_frame.pack()

        self.team_selector = tk.Listbox(listbox_frame, selectmode="multiple", exportselection=False, width=30, height=20)
        scrollbar = tk.Scrollbar(listbox_frame, orient="vertical", command=self.team_selector.yview)
        self.team_selector.config(yscrollcommand=scrollbar.set)
        self.team_selector.pack(side="left")
        scrollbar.pack(side="left", fill="y")

        # Group labels next to the team list
        self.group_labels = tk.Text(listbox_frame, height=20, width=10, state="disabled", bg="#f0f0f0", relief="flat")
        self.group_labels.pack(side="left", padx=(5, 0))

        self.count_label = ttk.Label(left_frame, text="Selected teams: 0/32")
        self.count_label.pack(pady=(5, 0))

        self.team_selector.bind('<<ListboxSelect>>', self.update_count_label)

        self.start_button = ttk.Button(left_frame, text="Start World Cup Simulation", command=self.run_simulation_thread)
        self.start_button.pack(pady=10)

        self.reset_button = ttk.Button(left_frame, text="Reset", command=self.reset_selection)
        self.reset_button.pack(pady=5)

        # Search field for teams
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(listbox_frame, textvariable=self.search_var, width=28)
        self.search_entry.pack(side="top", pady=(0, 5))
        self.search_var.trace("w", self.update_team_list)

        # === Right column: results and group info
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        self.result_text = tk.Text(right_frame, height=25, width=80)
        self.result_text.pack()

        self.group_info = tk.Text(right_frame, height=10, width=30, state="disabled", bg="#f8f8f8", relief="flat")
        self.group_info.pack(pady=(10, 0))

        self.sim_label = ttk.Label(right_frame, text="Number of simulations:")
        self.sim_label.pack()
        self.sim_entry = ttk.Entry(right_frame, width=10)
        self.sim_entry.insert(0, "100")
        self.sim_entry.pack()

        self.group_sim_label = ttk.Label(right_frame, text="Matches per group meeting:")
        self.group_sim_label.pack()
        self.group_sim_entry = ttk.Entry(right_frame, width=10)
        self.group_sim_entry.insert(0, "1")
        self.group_sim_entry.pack()

        # Fill team list for the first time
        self.all_teams = sorted(le_team.classes_)
        self.update_team_list()

    def update_team_list(self, *args):
        # Save currently selected teams (in the order they were selected)
        current_selection = [self.team_selector.get(i) for i in self.team_selector.curselection()]
        # Add new selections to self.selected_teams
        for team in current_selection:
            if team not in self.selected_teams and len(self.selected_teams) < 32:
                self.selected_teams.append(team)
        # Remove teams that were deselected
        self.selected_teams = [team for team in self.selected_teams if team in current_selection or team in self.selected_teams]

        search = self.search_var.get().lower()
        self.team_selector.delete(0, tk.END)
        for team in self.all_teams:
            if search in team.lower():
                self.team_selector.insert(tk.END, team)
        # Restore selection for teams that are still selected
        for idx in range(self.team_selector.size()):
            if self.team_selector.get(idx) in self.selected_teams:
                self.team_selector.selection_set(idx)

    def run_simulation_thread(self):
        # Run simulation in a separate thread to avoid freezing the GUI
        t = threading.Thread(target=self.run_simulation)
        t.start()

    def run_simulation(self):
        self.final_winners = Counter()
        try:
            n_sim = int(self.sim_entry.get())
        except Exception:
            n_sim = 100

        try:
            self.group_sim_count = int(self.group_sim_entry.get())
        except Exception:
            self.group_sim_count = 1

        if len(self.selected_teams) != 32:
            self.safe_insert("⚠️ Please select exactly 32 teams.\n")
            return

        self.safe_insert("🏁 Starting World Cup simulation...\n")

        group_winners = Counter()
        group_runners_up = Counter()

        for sim in range(n_sim):
            # 1. Group stage
            groups = [self.selected_teams[i:i+4] for i in range(0, 32, 4)]
            group_tables = {}
            for idx, group_teams in enumerate(groups):
                group_label = chr(65 + idx)
                table, match_stats = self.simulate_group(group_label, group_teams, self.group_sim_count)
                group_tables[group_label] = table
                if sim == n_sim - 1:
                    if 'match_stats_by_group' not in locals():
                        match_stats_by_group = {}
                    match_stats_by_group[group_label] = match_stats

            # 2. Get group winners and runners-up
            winners = {}
            runners_up = {}
            for group, table in group_tables.items():
                sorted_teams = table.sort_values(
                    by=["Pts", "GD", "GF"], ascending=[False, False, False]
                ).index.tolist()
                winners[group] = sorted_teams[0]
                runners_up[group] = sorted_teams[1]
                group_winners[sorted_teams[0]] += 1
                group_runners_up[sorted_teams[1]] += 1

            # 3. Knockout stage
            r16 = [
                (winners['A'], runners_up['B']),
                (winners['C'], runners_up['D']),
                (winners['E'], runners_up['F']),
                (winners['G'], runners_up['H']),
                (winners['B'], runners_up['A']),
                (winners['D'], runners_up['C']),
                (winners['F'], runners_up['E']),
                (winners['H'], runners_up['G']),
            ]
            qf = [self.sim_knockout_match(a, b) for a, b in r16]
            sf = [self.sim_knockout_match(qf[i], qf[i+1]) for i in range(0, 8, 2)]
            finalists = [self.sim_knockout_match(sf[0], sf[1]), self.sim_knockout_match(sf[2], sf[3])]
            winner = self.sim_knockout_match(finalists[0], finalists[1])
            self.final_winners[winner] += 1

            # Save last simulation for output
            if sim == n_sim - 1:
                last_group_tables = group_tables
                last_r16 = r16
                last_qf = qf
                last_sf = sf
                last_finalists = finalists
                last_final = winner

        # Output last simulation results
        self.result_text.delete("1.0", tk.END)
        self.safe_insert("\n📦 Group Stage:\n")
        for group, table in last_group_tables.items():
            self.safe_insert(f"\n--- Group {group} ---\n")
            self.safe_insert(table[["Pts", "W", "D", "L", "Wins"]].to_string())
            self.safe_insert("\n")
            # Print win statistics for each match
            if 'match_stats_by_group' in locals() and group in match_stats_by_group:
                for t1, t2, w1, w2, d, n_sim in match_stats_by_group[group]:
                    if w1 > w2:
                        winner = t1
                    elif w2 > w1:
                        winner = t2
                    else:
                        winner = "Draw"
                    self.safe_insert(f"{t1} vs {t2}: {t1} won {w1}/{n_sim}, {t2} won {w2}/{n_sim}, draws {d}/{n_sim} → Most: {winner}\n")

        self.safe_insert("\n🏟️ Knockout Stage:\n")
        self.safe_insert("\nAdvanced from groups:\n")
        for group, table in last_group_tables.items():
            sorted_teams = table.sort_values(
                by=["Pts", "GD", "GF"], ascending=[False, False, False]
            ).index.tolist()
            self.safe_insert(f"Group {group}: 1) {sorted_teams[0]}, 2) {sorted_teams[1]}\n")

        self.safe_insert("\nQuarterfinals:\n")
        for i in range(0, 8, 2):
            self.safe_insert(f"{last_r16[i][0]} vs {last_r16[i][1]} → {last_qf[i]}\n")
            self.safe_insert(f"{last_r16[i+1][0]} vs {last_r16[i+1][1]} → {last_qf[i+1]}\n")
        self.safe_insert("\nSemifinals:\n")
        self.safe_insert(f"{last_qf[0]} vs {last_qf[1]} → {last_sf[0]}\n")
        self.safe_insert(f"{last_qf[2]} vs {last_qf[3]} → {last_sf[1]}\n")
        self.safe_insert(f"{last_qf[4]} vs {last_qf[5]} → {last_sf[2]}\n")
        self.safe_insert(f"{last_qf[6]} vs {last_qf[7]} → {last_sf[3]}\n")
        self.safe_insert("\nFinal:\n")
        self.safe_insert(f"{last_sf[0]} vs {last_sf[1]} → {last_finalists[0]}\n")
        self.safe_insert(f"{last_sf[2]} vs {last_sf[3]} → {last_finalists[1]}\n")
        self.safe_insert(f"\n🏆 Final: {last_finalists[0]} vs {last_finalists[1]} → {last_final}\n")

        # Summary statistics
        self.safe_insert("\n🏆 Winners (all simulations):\n")
        for team, count in self.final_winners.most_common():
            self.safe_insert(f"{team}: {count} wins\n")

    def safe_insert(self, text):
        # Thread-safe insert into result_text
        self.after(0, lambda: self.result_text.insert(tk.END, text))
        self.after(0, self.result_text.see, tk.END)

    def simulate_group(self, label, teams, n_sim=1):
        # Simulate all group matches n_sim times each
        table = pd.DataFrame(index=teams, columns=["W", "D", "L", "Pts", "GD", "GF", "Wins"])
        table = table.fillna(0).infer_objects(copy=False)
        match_stats = []

        for i in range(4):
            for j in range(i+1, 4):
                t1, t2 = teams[i], teams[j]
                w1, w2, d = 0, 0, 0
                pts1, pts2 = 0, 0
                gf1, gf2 = 0, 0
                gd1, gd2 = 0, 0
                for _ in range(n_sim):
                    probs = self.predict_match(t1, t2)
                    outcome = np.random.choice(le_result.classes_, p=probs)
                    home_goals = np.random.poisson(1.5)
                    away_goals = np.random.poisson(1.2)
                    if outcome == "win":
                        w1 += 1
                        pts1 += 3
                        gf1 += home_goals
                        gf2 += away_goals
                        gd1 += home_goals - away_goals
                        gd2 += away_goals - home_goals
                    elif outcome == "loss":
                        w2 += 1
                        pts2 += 3
                        gf2 += away_goals
                        gf1 += home_goals
                        gd2 += away_goals - home_goals
                        gd1 += home_goals - away_goals
                    else:
                        d += 1
                        pts1 += 1
                        pts2 += 1
                        gf1 += home_goals
                        gf2 += away_goals
                        gd1 += home_goals - away_goals
                        gd2 += away_goals - home_goals
                # Update table
                table.loc[t1, "W"] += w1
                table.loc[t1, "D"] += d
                table.loc[t1, "L"] += w2
                table.loc[t1, "Pts"] += pts1
                table.loc[t1, "GF"] += gf1
                table.loc[t1, "GD"] += gd1
                table.loc[t1, "Wins"] += w1

                table.loc[t2, "W"] += w2
                table.loc[t2, "D"] += d
                table.loc[t2, "L"] += w1
                table.loc[t2, "Pts"] += pts2
                table.loc[t2, "GF"] += gf2
                table.loc[t2, "GD"] += gd2
                table.loc[t2, "Wins"] += w2

                # Save match statistics
                match_stats.append((t1, t2, w1, w2, d, n_sim))
        return table, match_stats

    def sim_knockout_match(self, t1, t2):
        # Simulate a knockout match, use penalty model if draw
        probs = self.predict_match(t1, t2)
        outcome = np.random.choice(le_result.classes_, p=probs)
        if outcome == "win":
            return t1
        elif outcome == "loss":
            return t2
        else:
            return self.predict_penalty(t1, t2)

    def predict_match(self, team1, team2):
        # Predict match outcome probabilities using the AI model
        if team1 not in le_team.classes_ or team2 not in le_team.classes_:
            return [1/3, 1/3, 1/3]
        df = pd.DataFrame([{
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
        return model.predict_proba(df)[0]

    def predict_penalty(self, t1, t2):
        # Predict penalty shootout winner using AI model, fallback to Elo if needed
        if t1 not in penalty_encoder.classes_ or t2 not in penalty_encoder.classes_:
            return t1 if get_elo(t1) >= get_elo(t2) else t2
        try:
            X = pd.DataFrame([{
                "home_team_enc": penalty_encoder.transform([t1])[0],
                "away_team_enc": penalty_encoder.transform([t2])[0],
                "first_shooter_enc": penalty_encoder.transform([t1])[0]
            }])
            pred = penalty_model.predict(X)[0]
            result = penalty_encoder.inverse_transform([pred])[0]
            return result if result in [t1, t2] else (t1 if get_elo(t1) >= get_elo(t2) else t2)
        except:
            return t1 if get_elo(t1) >= get_elo(t2) else t2

    def update_count_label(self, event=None):
        # Update selected teams and group info when selection changes
        current_selection = [self.team_selector.get(i) for i in self.team_selector.curselection()]
        for team in current_selection:
            if team not in self.selected_teams and len(self.selected_teams) < 32:
                self.selected_teams.append(team)
        self.selected_teams = [team for team in self.selected_teams if team in current_selection or team in self.selected_teams]
        if len(self.selected_teams) > 32:
            self.selected_teams = self.selected_teams[:32]
        count = len(self.selected_teams)
        self.count_label.config(text=f"Selected teams: {count}/32")
        self.update_group_labels()
        self.update_group_info()

    def update_group_labels(self):
        # Always show 32 rows, one for each possible team slot
        group_text = ""
        for i in range(32):
            group_idx = i // 4
            group_text += f"Group {chr(65+group_idx)}\n"
        self.group_labels.config(state="normal")
        self.group_labels.delete("1.0", tk.END)
        self.group_labels.insert(tk.END, group_text)
        self.group_labels.config(state="disabled")

    def update_group_info(self):
        # Show which teams are in which group
        group_text = ""
        for group_idx in range(8):
            group_lag = self.selected_teams[group_idx*4:(group_idx+1)*4]
            if group_lag:
                group_text += f"Group {chr(65+group_idx)}: " + ", ".join(group_lag) + "\n"
        self.group_info.config(state="normal")
        self.group_info.delete("1.0", tk.END)
        self.group_info.insert(tk.END, group_text)
        self.group_info.config(state="disabled")

    def reset_selection(self):
        # Deselect all teams and reset group info
        self.team_selector.selection_clear(0, tk.END)
        self.selected_teams = []
        self.update_count_label()
        self.group_info.config(state="normal")
        self.group_info.delete("1.0", tk.END)
        self.group_info.config(state="disabled")

if __name__ == "__main__":
    app = VMSimulatorApp()
    app.mainloop()