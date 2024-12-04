import threading
from project_package import testcsvconnection, naivebayesclassifier, randomforestclassifier

'''
Collaborators:
Linu Robin
Stephanie Scherb
Aaron Block

This project is for determining fraudulent transaction activity 
by using Naive Bayes and Random Forest classifier models for 
CPSC_4370_AI's end of the semester project

The referenced datasets being used for training and testing each model are Train.csv and Test.csv
These datasets may be found at https://www.kaggle.com/datasets/kartik2112/fraud-detection
All resources were discovered through research.
While the datasets are open source, all code has been written and committed by the students/collaborators 
for this project. For more information, refer to https://github.com/blockkaaron/AI_Project_CPSC_4370

To run program, ensure train.csv and test.csv are in project as they are the datasets being used
Please give the program enough time to complete each step of both models (approximately 1 minute)

There will be 4 pop-up results in this order: 

Pre-Data Prep
Post-Data Prep
Naive Bayes Confusion Matrix 
Random Forest Confusion Matrix

The console will also print results based on progress of each model.
'''

def run_models():
    testcsvconnection
    naivebayesclassifier
    randomforestclassifier

thread = threading.Thread(target = run_models)

thread.start()

thread.join()