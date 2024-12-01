import pandas as pd

'''
This .py is to test your file structure to ensure you're ready to run main.py
'''
# attempt to test connection to data file
try:
    # here, we establish our data or file path and read file
    datainput = pd.read_csv('creditcard.csv')

    # print first 10 rows to test
    print(datainput[0:10],)
    print(datainput.shape,"Total columns/rows read successfully")

# catch exception
except:
    print("Exception occurred, \nPlease check the creditcard.csv file location and ensure it is accessible.")