import requests 
import time
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

API_key = os.getenv('AVIATION_API_KEY')

if not API_key:
    raise ValueError("Not found API Key. Let's check file .env!")

url = f"http://api.aviationstack.com/v1/flights?access_key={API_key}"
total = 3000
limit = 100
final_data = []

print(f'Start to collect {total} line of data')

for current_offset in range(0,total,limit):
    params = {'access_key': API_key,
              'flight_status' : 'landed',
            'limit': limit,
        'offset': current_offset}
    
    try:
        r = requests.get(url,params= params)
        if r.status_code == 200:
            data = r.json()
            batch = data.get('data',[])
            if not batch:
                print('There is no more data left to retrieve!')
                break
            final_data.extend(batch)
            print(f"We collected: {len(final_data)}/{total}")
        else:
            print(f"Error at offset {current_offset}: {r.status_code}")
    except Exception as e:
        print(f'An error occurred: {e}')
    time.sleep(1)

df = pd.json_normalize(final_data)
df.to_csv('D:/Full projet/EcoFlight_Delay_Predictor/data/raw.csv', index=False)  
print('Completed to save !')