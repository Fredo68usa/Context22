import pandas as pd

# Create a dataframe
df = pd.DataFrame({'Name': ['John', 'Mary', 'Bob'], 'Age': [20, 25, 30]})
print(df)

# Create a series
s = pd.Series([1, 2, 3], name='New Column')
print(ss)

# Add the series to the dataframe using the assign() function
df = df.assign(New Column=s)

# Print the dataframe
print(df)
