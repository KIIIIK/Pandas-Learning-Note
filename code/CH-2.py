import pandas as pd

s = pd.Series(['banana', 42])
print(s)

s = pd.Series(['Wes McKinney', 'Creator of Pandas'],
                index=['Person', 'Who'])
print(s)


scientists = pd.DataFrame({
    'Name': ['Rosaline Franklin', 'William Gosset'],
    'Occupation': ['Chemist', 'Statistician'],
    'Born': ['1920-07-25', '1876-06-13'],
    'Died': ['1958-04-16', '1937-10-16'],
    'Age': [37, 61]
})
print(scientists)















