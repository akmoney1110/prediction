from playwright.sync_api import sync_playwright
from datetime import datetime, timedelta
from predict.models import Fixture
import pytz

def scrape_fixtures_with_league_code(url, league_code):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36")
        page.goto(url, timeout=60000)
        page.wait_for_load_state("networkidle")
        page.wait_for_selector("div.sc-8b63db7c-3", timeout=30000)

        fixtures = []
        fixture_elements = page.query_selector_all("div.sc-8b63db7c-3")

        for fixture in fixture_elements:
            try:
                # Get home and away teams
                home = fixture.query_selector("span.visibleSM").inner_text().strip()
                away = fixture.query_selector_all("span.visibleSM")[1].inner_text().strip()

                # Get time string (like '20:00')
                time_str = fixture.query_selector("span.sc-4e4c9eab-2.scheduled").inner_text().strip()

                fixtures.append({
                    'league_code': league_code,
                    'home': home,
                    'away': away,
                    'time_str': time_str
                })
            except Exception as e:
                print(f"⚠️ Skipping one fixture due to error: {e}")

        browser.close()
        return fixtures

def save_fixtures(fixtures, timezone='Europe/London'):
    tz = pytz.timezone(timezone)
    today = datetime.now(tz).date()

    for f in fixtures:
        # Parse the time string assuming HH:MM and today as date
        try:
            time_obj = datetime.strptime(f['time_str'], '%H:%M').time()
        except ValueError:
            print(f"⚠️ Invalid time format for fixture: {f}")
            continue

        match_datetime = datetime.combine(today, time_obj)
        match_datetime = tz.localize(match_datetime)

        # Save or update fixture
        obj, created = Fixture.objects.update_or_create(
            league_code=f['league_code'],
            home_team=f['home'],
            away_team=f['away'],
            match_time=match_datetime,
            defaults={}
        )
        action = "Created" if created else "Updated"
        print(f"✅ {action} fixture: {obj}")

def scrape_all_leagues_and_save():
    league_info = {
        'E0': 'https://int.soccerway.com/national/england/premier-league/2025-2026/regular-season/859443d4-5504-4dca-8609-624cbfb592ad/',
        'E1': 'https://int.soccerway.com/national/england/championship/2025-2026/regular-season/xxxxxx/',  # replace with actual URL
        # Add more leagues here...
    }

    all_fixtures = []
    for league_code, url in league_info.items():
        print(f"🕵️ Scraping fixtures for league {league_code}...")
        fixtures = scrape_fixtures_with_league_code(url, league_code)
        print(f"Found {len(fixtures)} fixtures for {league_code}")
        all_fixtures.extend(fixtures)

    save_fixtures(all_fixtures)
    print("🎉 All fixtures saved/updated successfully.")
