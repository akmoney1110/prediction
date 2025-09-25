import requests
import pandas as pd
import os
from datetime import datetime
from hashlib import md5
import pandas as pd
import os
import csv  # ✅ Add this
import subprocess

API_KEY = "63ebf94991a895238ad140325b00759e"
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}
OUTPUT_FILE = "csv_data/merged_full_football_data.csv"

LEAGUE_CODE_MAP = {
    "Premier League": "E0",
    "Championship": "E1",
    "Bundesliga 1": "D1",
    "La Liga": "SP1",
    "Serie A": "I1",
    "Ligue 1": "F1",
    "UEFA Champions League": "CL",
    "UEFA Europa League": "EL",
    "FIFA World Cup": "WC",
    "Africa Cup of Nations": "AFCON",
    "Copa America": "CA",
    "UEFA Euro Championship": "EURO",
    # Add more as needed
}

def get_fixtures(date_str):
    url = f"{BASE_URL}/fixtures"
    params = {"date": date_str, "status": "FT"}
    res = requests.get(url, headers=HEADERS, params=params)
    return res.json().get("response", [])

def get_statistics(fixture_id):
    url = f"{BASE_URL}/fixtures/statistics"
    res = requests.get(url, headers=HEADERS, params={"fixture": fixture_id})
    stats = res.json().get("response", [])
    return {item['team']['name']: {s['type']: s['value'] for s in item['statistics']} for item in stats}

def get_odds(fixture_id):
    url = f"{BASE_URL}/odds"
    res = requests.get(url, headers=HEADERS, params={"fixture": fixture_id})
    try:
        odds_data = res.json()['response'][0]['bookmakers'][0]['bets']
        odds_dict = {}
        for bet in odds_data:
            for val in bet['values']:
                odds_dict[f"{bet['name']}_{val['value']}"] = val['odd']
        return odds_dict
    except (IndexError, KeyError):
        return {}

def generate_match_key(date, home, away):
    return md5(f"{date}_{home}_{away}".encode()).hexdigest()

def extract_features(fixture, stats, odds):
    home_team = fixture['teams']['home']['name']
    away_team = fixture['teams']['away']['name']
    goals = fixture['goals']
    league = fixture['league']['name']

    div = LEAGUE_CODE_MAP.get(league, league[:3].upper())

    row = {
        "Div": div,
        "Date": fixture['fixture']['date'][:10],
        "Time": fixture['fixture']['date'][11:16],
        "HomeTeam": home_team,
        "AwayTeam": away_team,
        "FTHG": goals['home'],
        "FTAG": goals['away'],
        "HTHG": fixture.get('score', {}).get('halftime', {}).get('home', 0),
        "HTAG": fixture.get('score', {}).get('halftime', {}).get('away', 0),
        "HTR": None,  # You can compute if needed
        "HomeWin": int(goals['home'] > goals['away']),
    }

    stat_fields = ['Shots on Goal', 'Total Shots', 'Fouls', 'Corner Kicks', 'Yellow Cards', 'Red Cards']
    for side, prefix in [(home_team, "H"), (away_team, "A")]:
        for field in stat_fields:
            key = f"{prefix}_{field.replace(' ', '')}"
            row[key] = stats.get(side, {}).get(field, 0)

    for key, val in odds.items():
        row[key.replace(" ", "_")] = val

    row["match_id"] = generate_match_key(row["Date"], home_team, away_team)
    return row

def append_to_csv(data, output_file=OUTPUT_FILE):
    df_new = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    

    def safe_read_csv(file_path):
        rows = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            for i, row in enumerate(reader, start=2):
                if len(row) == len(header):
                    rows.append(row)
                else:
                    print(f"⚠️ Skipping bad row {i}: {len(row)} fields (expected {len(header)})")
        return pd.DataFrame(rows, columns=header)


    if os.path.exists(output_file):
        df_existing = safe_read_csv(output_file)
        existing_ids = set(df_existing.get("match_id", []))
        df_new = df_new[~df_new["match_id"].isin(existing_ids)]
        df_final = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_final = df_new

    df_final.to_csv(output_file, index=False)
    print(f"✅ Appended {len(df_new)} new match(es). Total: {len(df_final)}")


def main():
    today = datetime.today().strftime("%Y-%m-%d")
    print(f"📅 Fetching completed matches for: {today}")
    fixtures = get_fixtures(today)

    all_rows = []
    for match in fixtures:
        fixture_id = match['fixture']['id']
        try:
            stats = get_statistics(fixture_id)
            odds = get_odds(fixture_id)
            row = extract_features(match, stats, odds)
            all_rows.append(row)
            print(f"✅ Processed: {row['HomeTeam']} vs {row['AwayTeam']}")
        except Exception as e:
            print(f"⚠️ Skipped fixture {fixture_id}: {e}")

    if all_rows:
        append_to_csv(all_rows)
    else:
        print("⚠️ No completed matches or data found.")



def retrain_model():
    print("🔁 Retraining prediction model...")
    subprocess.run(["python", "manage.py", "train_predictor"])
    print("✅ Model retrained.")        

if __name__ == "__main__":
    main()



retrain_model()