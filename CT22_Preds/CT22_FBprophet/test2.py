import pandas as pd
from datetime import datetime

# df = pd.DataFrame({'date': ['2023-03-08', '2023-03-09', '2023-03-10']})
df = pd.DataFrame({'date': ['2023-03-08 12:00:01', '2023-03-09 13:01:00', '2023-03-10 14:00:00']})
df['date'] = pd.to_datetime(df['date'])
df['timestamp'] = df['date'].astype('int64') // 10**9
print(df)

