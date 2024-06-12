import pandas as pd
df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
print (df)
df.rename(columns={"A": "a", "B": "c"}, inplace=True)
print (df)
