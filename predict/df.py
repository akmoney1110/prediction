import requests

API_KEY = '6c5bcbd4223dd1407b92d768d451af75'

import requests

API_KEY = "6c5bcbd4223dd1407b92d768d451af75"
BASE_URL = "https://v3.football.api-sports.io"

headers = {
    'x-apisports-key': API_KEY
}

response = requests.get(f"{BASE_URL}/fixtures?live=all", headers=headers)

if response.status_code == 200:
    print(response.json())
else:
    print("Error:", response.status_code, response.text)
