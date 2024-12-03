import pandas as pd

'''
This .py is to test your file structure to ensure you're ready to run main.py
Collaborators:
Linu Robin
Stephanie Scherb
Aaron Block
'''
# attempt to test connection to data files
try:
    # here, we establish our data files and read them in to confirm connection
    testdata = pd.read_csv('../test.csv')
    traindata = pd.read_csv('../train.csv')

    # print first 10 rows to test
    print(testdata[0:10],)
    print(testdata.shape,"Total columns/rows from test.csv read successfully")
    print("\n")
    print(traindata[0:10],)
    print(traindata.shape,"Total columns/rows from train.csv read successfully")
    print("\n")
    print("Executing Naive Bayes classifier model, please wait...")

# catch exceptions
except:
    print("Exception occurred, \nPlease check the .CSV files, their locations, and ensure they are accessible.")