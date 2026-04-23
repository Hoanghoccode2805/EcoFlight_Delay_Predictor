import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler

# 1. Using raw string or relative path for portability
raw_path = r'D:/Full projet/EcoFlight_Delay_Predictor/data/raw.csv'
raw_df = pd.read_csv(raw_path)

# 2. Check the infomation of raw data
print(raw_df.head())
print(raw_df.info())
print('\n Shape:')
print(raw_df.shape)
print(raw_df.describe())

missing_pct = 100*(raw_df.notna().sum()/len(raw_df))
print(missing_pct)

# 3. Keep only features relevant to flight efficiency and operational delay
keep_cols = ['airline.name', 'departure.airport', 'arrival.airport',
           'departure.delay', 'departure.scheduled','arrival.scheduled', 'arrival.delay','departure.actual']
df = raw_df[keep_cols].copy()

# 4. Remove records missing critical identification data
df = df.dropna(subset = ['airline.name','departure.airport','arrival.airport'])
df = df.drop_duplicates(subset = ['airline.name','departure.airport','arrival.airport', 'departure.scheduled'])

# 5. Datatime conversion: Convert string timestamps to datetime objects for calculation
time_cols = ['departure.scheduled', 'departure.actual', 'arrival.scheduled']
for col in time_cols:
    df[col] = pd.to_datetime(df[col])

# 6. Manually calculate departure delay to avoid API data inconsistencies
# Result in minutes: Positive = Late, Negative = Early
df['departure.delay'] = (df['departure.actual'] - df['departure.scheduled']).dt.total_seconds()/60

# 7. Filter out unrealistic data: flights departing >1h early or >12h late
df = df[(df['departure.delay'] > -60) & (df['departure.delay'] < 720)]

# 8. Fill the missing data of delay
df['departure.delay'] = df['departure.delay'].fillna(0)
df['arrival.delay'] = df['arrival.delay'].fillna(0)

# 9. Extract time-based features: Hour of departure
df['departure_hour'] = df['departure.scheduled'].dt.hour

# 10. Calculate scheduled flight duration
df['scheduled_duration'] = (df['arrival.scheduled'] - df['departure.scheduled']).dt.total_seconds() / 60
df = df[df['scheduled_duration'] > 0]

# 11. Standard Aviation Industry Threshold: 15 minutes
df['is_delayed'] = (df['arrival.delay'] > 15).astype(int)

# 12. Transform text-based categories (Airlines/Airports) into numerical labels for the model
le = LabelEncoder()
cat_features = ['airline.name', 'departure.airport', 'arrival.airport']
for col in cat_features:
    df[col] = le.fit_transform(df[col].astype(str))

# 13. Normalize numerical features to improve gradient descent convergence
scaler = StandardScaler()
num_features = ['departure.delay', 'departure_hour', 'scheduled_duration']
df[num_features] = scaler.fit_transform(df[num_features])

# 14. Final Feature Selection & Export: Select finalized features for training and export to CSV
final_cols = cat_features + num_features + ['is_delayed']
df_final = df[final_cols].copy()

# 15. Check the processed data
print(df_final.info)
missing = 100*(df_final.notna().sum()/len(df_final))
print(missing)

# 16. Save data 
output_path = r'D:/Full projet/EcoFlight_Delay_Predictor/data/process.csv'
df_final.to_csv(output_path, index=False)       

print(f"\nSuccessfully saved processed data to: {output_path}")
