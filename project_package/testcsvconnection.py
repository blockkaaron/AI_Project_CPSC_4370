import pandas as pd

'''
This .py is to test your file structure to ensure you're ready to run main.py
'''
# attempt to test connection to data files
try:
    # here we establish our data files and read them in to confirm connection
    datatest = pd.read_csv('../test.csv')
    datatrain = pd.read_csv('../train.csv')

    # print first 10 rows to test
    print(datatest[0:10],)
    print(datatest.shape,"Total columns/rows from test.csv read successfully")
    print("\n")
    print(datatrain[0:10],)
    print(datatrain.shape,"Total columns/rows from train.csv read successfully")

# catch exceptions
except:
    print("Exception occurred, \nPlease check the .CSV files, their locations, and ensure they are accessible.")