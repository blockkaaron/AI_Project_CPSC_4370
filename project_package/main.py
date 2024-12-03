import threading
from project_package import naivebayesclassifier, randomforestclassifier

'''
Determining fraudulent transaction activity by using Naive Bayes and Random Forest Classifier models

To run, ensure train.csv and test.csv are in project as they are the datasets being used
There will be 2 pop-ups: Naive Bayes Confusion Matrix and Random Forest Classifier Confusion Matrix
'''

def run_models():
    naivebayesclassifier
    randomforestclassifier

thread = threading.Thread(target=run_models)

thread.start()

thread.join()