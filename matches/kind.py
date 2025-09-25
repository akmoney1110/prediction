import http.client
import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import seaborn as sns
from tabulate import tabulate


class FootballAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.host = "v3.football.api-sports.io"
        self.headers = {
            'x-rapidapi-host': self.host,
            'x-rapidapi-key': self.api_key
        }
        self.conn = None

    def connect(self):
        """Establish connection to the API"""
        try:
            self.conn = http.client.HTTPSConnection(self.host)
            return True
        except Exception as e:
            print(f"Connection error: {e}")
            return False

    def get_standings(self, league, season):
        """Get standings for a specific league and season"""
        if not self.conn:
            if not self.connect():
                return None

        try:
            endpoint = f"/standings?league={league}&season={season}"
            self.conn.request("GET", endpoint, headers=self.headers)
            res = self.conn.getresponse()

            if res.status == 200:
                data = res.read()
                return json.loads(data.decode("utf-8"))
            else:
                print(f"API Error: {res.status} - {res.reason}")
                return None

        except Exception as e:
            print(f"Request error: {e}")
            return None

    def close(self):
        """Close the connection"""
        if self.conn:
            self.conn.close()

    def process_standings_data(self, data):
        """Process and extract standings data"""
        if not data or 'response' not in data:
            return None

        try:
            standings = data['response'][0]['league']['standings'][0]
            league_info = data['response'][0]['league']
            return standings, league_info
        except (KeyError, IndexError) as e:
            print(f"Data processing error: {e}")
            return None, None

    def display_standings_table(self, standings, league_info):
        """Display standings in a formatted table"""
        if not standings:
            return

        table_data = []
        for team in standings:
            table_data.append([
                team['rank'],
                team['team']['name'],
                team['all']['played'],
                team['all']['win'],
                team['all']['draw'],
                team['all']['lose'],
                team['all']['goals']['for'],
                team['all']['goals']['against'],
                team['goalsDiff'],
                team['points'],
                team['form']
            ])

        headers = ["Pos", "Team", "MP", "W", "D", "L", "GF", "GA", "GD", "Pts", "Form"]
        print(f"\n{league_info['name']} {league_info['season']} Standings")
        print("=" * 80)
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

    def create_standings_visualization(self, standings, league_info):
        """Create visualization of the standings data"""
        if not standings:
            return

        # Prepare data for visualization
        teams = [team['team']['name'] for team in standings]
        points = [team['points'] for team in standings]
        goals_for = [team['all']['goals']['for'] for team in standings]
        goals_against = [team['all']['goals']['against'] for team in standings]

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        fig.suptitle(f"{league_info['name']} {league_info['season']} Standings", fontsize=16, fontweight='bold')

        # Points bar chart
        colors = ['#1f77b4' if i < 4 else ('#ff7f0e' if i < 6 else '#2ca02c') for i in range(len(teams))]
        ax1.barh(teams, points, color=colors)
        ax1.set_xlabel('Points')
        ax1.set_title('League Points')
        ax1.invert_yaxis()  # Highest points at top

        # Goals scatter plot
        ax2.scatter(goals_for, goals_against, s=100, alpha=0.7)
        ax2.set_xlabel('Goals For')
        ax2.set_ylabel('Goals Against')
        ax2.set_title('Goals Analysis')

        # Add team labels to scatter plot
        for i, team in enumerate(teams):
            ax2.annotate(team[:15], (goals_for[i], goals_against[i]),
                         xytext=(5, 5), textcoords='offset points', fontsize=8)

        # Add a reference line for goal difference
        max_goals = max(max(goals_for), max(goals_against))
        ax2.plot([0, max_goals], [0, max_goals], 'r--', alpha=0.5)

        plt.tight_layout()
        plt.savefig('standings_visualization.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved as 'standings_visualization.png'")

    def save_standings_to_csv(self, standings, league_info):
        """Save standings data to CSV file"""
        if not standings:
            return

        data = []
        for team in standings:
            data.append({
                'Position': team['rank'],
                'Team': team['team']['name'],
                'Played': team['all']['played'],
                'Won': team['all']['win'],
                'Drawn': team['all']['draw'],
                'Lost': team['all']['lose'],
                'Goals For': team['all']['goals']['for'],
                'Goals Against': team['all']['goals']['against'],
                'Goal Difference': team['goalsDiff'],
                'Points': team['points'],
                'Form': team['form']
            })

        df = pd.DataFrame(data)
        filename = f"{league_info['name'].replace(' ', '_')}_{league_info['season']}_standings.csv"
        df.to_csv(filename, index=False)
        print(f"Data saved to '{filename}'")

    def get_team_stats(self, standings, team_name):
        """Get specific team statistics"""
        if not standings:
            return None

        for team in standings:
            if team['team']['name'].lower() == team_name.lower():
                return team

        return None


def main():
    # Initialize the API client
    api_key = "6c5bcbd4223dd1407b92d768d451af75"  # Consider using environment variables for security
    football_api = FootballAPI(api_key)

    # Get standings for Premier League (ID: 39) for 2019 season
    print("Fetching Premier League 2019 standings...")
    data = football_api.get_standings(39, 2019)

    if data:
        standings, league_info = football_api.process_standings_data(data)

        if standings:
            # Display formatted table
            football_api.display_standings_table(standings, league_info)

            # Create visualization
            football_api.create_standings_visualization(standings, league_info)

            # Save to CSV
            football_api.save_standings_to_csv(standings, league_info)

            # Example: Get specific team stats
            team_name = "Liverpool"
            team_stats = football_api.get_team_stats(standings, team_name)
            if team_stats:
                print(f"\n{team_name} Stats:")
                print(f"Position: {team_stats['rank']}")
                print(f"Points: {team_stats['points']}")
                print(f"Form: {team_stats['form']}")
            else:
                print(f"\nTeam '{team_name}' not found in standings")
        else:
            print("Failed to process standings data")
    else:
        print("Failed to fetch data from API")

    # Close connection
    football_api.close()


if __name__ == "__main__":
    main()
