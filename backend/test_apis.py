"""Quick test script to verify API connections."""

import asyncio
import httpx


async def test_odds_api():
    """Test The Odds API connection."""
    api_key = "b1f7f077c2189762410f65bcb6414fab"

    print("=" * 50)
    print("Testing The Odds API")
    print("=" * 50)

    async with httpx.AsyncClient() as client:
        # Test events endpoint
        response = await client.get(
            "https://api.the-odds-api.com/v4/sports/basketball_nba/events",
            params={"apiKey": api_key},
            timeout=30.0,
        )

        if response.status_code == 200:
            games = response.json()
            print(f"SUCCESS: Found {len(games)} upcoming NBA games")

            remaining = response.headers.get("x-requests-remaining")
            used = response.headers.get("x-requests-used")
            print(f"API Usage: {used} used, {remaining} remaining")

            if games:
                print("\nUpcoming games:")
                for game in games[:5]:
                    print(f"  {game['away_team']} @ {game['home_team']}")
                    print(f"    Time: {game['commence_time']}")
        else:
            print(f"ERROR: Status {response.status_code}")
            print(response.text)

        # Test odds endpoint
        print("\nFetching odds...")
        response = await client.get(
            "https://api.the-odds-api.com/v4/sports/basketball_nba/odds",
            params={
                "apiKey": api_key,
                "regions": "us",
                "markets": "h2h,spreads,totals",
                "oddsFormat": "decimal",
            },
            timeout=30.0,
        )

        if response.status_code == 200:
            odds_data = response.json()
            print(f"SUCCESS: Got odds for {len(odds_data)} games")

            if odds_data:
                game = odds_data[0]
                print(f"\nSample game: {game['away_team']} @ {game['home_team']}")
                print(f"Bookmakers: {len(game.get('bookmakers', []))}")

                for bm in game.get("bookmakers", [])[:2]:
                    print(f"\n  {bm['key']}:")
                    for market in bm.get("markets", []):
                        print(f"    {market['key']}:")
                        for outcome in market.get("outcomes", []):
                            line = f" ({outcome.get('point')})" if outcome.get('point') else ""
                            print(f"      {outcome['name']}{line}: {outcome['price']}")
        else:
            print(f"ERROR: Status {response.status_code}")


async def test_balldontlie():
    """Test BALLDONTLIE API connection."""
    api_key = "d9441240-7336-41b9-bf9c-ef664de48b9d"

    print("\n" + "=" * 50)
    print("Testing BALLDONTLIE API")
    print("=" * 50)

    headers = {"Authorization": api_key}

    async with httpx.AsyncClient() as client:
        # Test teams endpoint
        response = await client.get(
            "https://api.balldontlie.io/v1/teams",
            headers=headers,
            timeout=30.0,
        )

        if response.status_code == 200:
            data = response.json()
            teams = data.get("data", [])
            print(f"SUCCESS: Found {len(teams)} NBA teams")

            if teams:
                print("\nSample teams:")
                for team in teams[:5]:
                    print(f"  {team['full_name']} ({team['abbreviation']})")
        else:
            print(f"ERROR: Status {response.status_code}")
            print(response.text)

        # Test games endpoint
        print("\nFetching today's games...")
        from datetime import date
        today = date.today().isoformat()

        response = await client.get(
            "https://api.balldontlie.io/v1/games",
            headers=headers,
            params={"start_date": today, "end_date": today},
            timeout=30.0,
        )

        if response.status_code == 200:
            data = response.json()
            games = data.get("data", [])
            print(f"SUCCESS: Found {len(games)} games today")

            for game in games:
                home = game["home_team"]["abbreviation"]
                away = game["visitor_team"]["abbreviation"]
                status = game["status"]
                print(f"  {away} @ {home} - {status}")
        else:
            print(f"ERROR: Status {response.status_code}")

        # Test injuries endpoint (requires paid tier)
        print("\nTesting injuries endpoint...")
        response = await client.get(
            "https://api.balldontlie.io/v1/injuries",
            headers=headers,
            timeout=30.0,
        )

        if response.status_code == 200:
            data = response.json()
            injuries = data.get("data", [])
            print(f"SUCCESS: Found {len(injuries)} injuries")

            for inj in injuries[:5]:
                player = inj.get("player", {})
                name = f"{player.get('first_name', '')} {player.get('last_name', '')}"
                status = inj.get("status", "Unknown")
                print(f"  {name}: {status}")
        elif response.status_code == 403:
            print("INFO: Injuries endpoint requires paid tier (expected)")
        else:
            print(f"ERROR: Status {response.status_code}")


async def main():
    await test_odds_api()
    await test_balldontlie()
    print("\n" + "=" * 50)
    print("API Tests Complete!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
