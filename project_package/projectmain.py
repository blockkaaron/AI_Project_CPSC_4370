import pandas as pd

datainput = pd.read_csv('../creditcard.csv')

print(datainput[0:5],"\n")

print("Shape of Complete Data Set")
print(datainput.shape,"\n")