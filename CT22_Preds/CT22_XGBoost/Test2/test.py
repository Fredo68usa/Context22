import pandas as pd

# Create a dataframe
df = pd.DataFrame({'Name': ['John', 'Mary', 'Bob'], 'Age': [20, 25, 30]})
print(df)

# Create a series
s = pd.Series([1, 2, 3], name='New Column')
print(s)

# Add the series to the dataframe using the insert() function
# df.insert(2, 'New Column', s)
df = df.assign(F = s)

# Print the dataframe
print(df)
