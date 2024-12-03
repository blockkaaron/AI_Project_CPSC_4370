import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

'''
Naive Bayes Classifier - Determining fraudulent transaction activity
Collaborators:
Linu Robin
Stephanie Scherb
Aaron Block
'''

# First we need to load and prepare the datasets
traindata = pd.read_csv("../train.csv")
testdata = pd.read_csv("../test.csv")
combineddata = pd.concat([traindata, testdata])
combineddata.head()
combineddata.info()
completedata = combineddata.dropna()
completedata.isnull().sum()

# here we identify duplicates and print result
completedata.duplicated().sum()
completedata.info()

# process data
fraud = combineddata[combineddata["is_fraud"] == 1]
notfraud = combineddata[combineddata["is_fraud"] == 0]
print(fraud.shape[0])
print(notfraud.shape[0])

notfraud = notfraud.sample(fraud.shape[0])
data = pd.concat([fraud,notfraud])

# drop unnecessary columns from dataset and print result to console
removecolumns = ['Unnamed: 0','first','last','unix_time','street','gender','job','dob','city','state','trans_num','merchant']
data.drop(columns = removecolumns, inplace = True)
data.info()
data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
data['trans_day'] = data['trans_date_trans_time'].dt.day
data['trans_month'] = data['trans_date_trans_time'].dt.month
data['trans_year'] = data['trans_date_trans_time'].dt.year
data['trans_hour'] = data['trans_date_trans_time'].dt.hour
data['trans_minute'] = data['trans_date_trans_time'].dt.minute
data.drop(columns = ['trans_date_trans_time'], inplace = True)

# encode category and cc_num, print result, then make features more comparable via feature scaling
encoder = LabelEncoder()
data['category'] = encoder.fit_transform(data['category'])
data['cc_num'] = encoder.fit_transform(data['cc_num'])
data.head()
scale = StandardScaler()
data['amt'] = scale.fit_transform(data[['amt']])
data['zip'] = scale.fit_transform(data[['zip']])
data['city_pop'] = scale.fit_transform(data[['city_pop']])
data['cc_num'] = encoder.fit_transform(data['cc_num'])

# here, we will split the data for accuracy
X = data.drop('is_fraud',axis=1)
y = data['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# here, we begin to train the Random Forest classifier
randomforest = RandomForestClassifier(random_state = 0)
randomforest.fit(X_train, y_train)
y_pred = randomforest.predict(X_test)

# make predictions/assumptions on the test set and print results for evaluation
print('Accuracy:', accuracy_score(y_test, y_pred))
print('F1 score:', f1_score(y_test, y_pred))
print('Confusion matrix:', confusion_matrix(y_test, y_pred))

# generate the confusion matrix result for model
labels = ["Not Fraud", "Fraud"]
cmatrix = confusion_matrix(y_test, y_pred)
display = ConfusionMatrixDisplay(confusion_matrix = cmatrix, display_labels = labels)
display.plot()
plt.suptitle('RFC Confusion Matrix', fontsize = 16, fontweight = 'bold')
mytitle = ("Random Forest model accuracy(in %):", metrics.accuracy_score(y_test, y_pred) * 100)
print(mytitle, type(mytitle))
s = str(mytitle)
print(s, type(s))
plt.title(s)
plt.show()

# comparing actual response values (test) with predicted response values (pred)
print("Random Forest Classifier model accuracy(in %):", metrics.accuracy_score(y_test, y_pred) * 100)
print("\n")
print("Project has completed Fraudulent Transaction Activity Prediction using Naive Bayes and Random Forest classifier models.")