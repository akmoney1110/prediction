# fetch_fixtures.py

import requests
import os
from datetime import datetime, timedelta
import pandas as pd




API_KEY = "a59a75941136b1ea23f40ab7fe32c4a2"
BASE_URL = "https://v3.football.api-sports.io"

HEADERS = {
    "x-apisports-key": API_KEY
}

# Map display names to API league IDs (you'll need to confirm these from API-Football)
TARGET_LEAGUES = {
    "E0": 39,    # Premier League
    "E1": 40,    # Championship
    "D1": 78,    # Bundesliga
    "SP1": 140,  # La Liga
    "I1": 135,   # Serie A
    "F1": 61,    # Ligue 1
    "CL": 2,     # Champions League
    "EL": 3,     # Europa League
    "WC": 1,     # World Cup
    "AFCON": 6,  # Africa Cup of Nations
    "CA": 9,     # Copa America
    
}

def fetch_upcoming_fixtures(days_ahead=3000):
    today = datetime.today().date()
    end_date = today + timedelta(days=days_ahead)
    all_fixtures = []

    for div, league_id in TARGET_LEAGUES.items():
        print(f"📡 Fetching fixtures for {div} (league_id={league_id})...")

        for day in range(days_ahead):
            target_date = (today + timedelta(days=day)).strftime("%Y-%m-%d")
            url = f"{BASE_URL}/fixtures"
            params = {
                "league": league_id,
                "season": datetime.today().year,
                "date": target_date
            }

            response = requests.get(url, headers=HEADERS, params=params)
            if response.status_code != 200:
                print(f"❌ Failed for {div} on {target_date}")
                continue

            data = response.json().get("response", [])
            for match in data:
                fixture = match["fixture"]
                teams = match["teams"]

                all_fixtures.append({
                    "Div": div,
                    "Date": fixture["date"][:10],
                    "Time": fixture["date"][11:16],
                    "HomeTeam": teams["home"]["name"],
                    "AwayTeam": teams["away"]["name"],
                    "Venue": fixture.get("venue", {}).get("name", ""),
                    "City": fixture.get("venue", {}).get("city", ""),
                    "MatchId": fixture["id"]
                })

    df = pd.DataFrame(all_fixtures)
    return df

def save_fixtures_csv(df, filename="upcoming_fixtures.csv"):
    if df.empty:
        print("⚠️ No upcoming fixtures found.")
        return
    df.to_csv(filename, index=False)
    print(f"✅ Saved {len(df)} upcoming fixtures to {filename}")

if __name__ == "__main__":
    fixtures_df = fetch_upcoming_fixtures(days_ahead=7)
    save_fixtures_csv(fixtures_df)
