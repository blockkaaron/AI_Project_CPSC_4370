import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

'''
Random Forest Classifier - Determining fraudulent transaction activity
'''

# First we need to load and prepare the datasets
data_train = pd.read_csv("../train.csv")
data_test = pd.read_csv("../test.csv")
full_data = pd.concat([data_train, data_test])
full_data.head()
full_data.info()
full_data = full_data.dropna()
full_data.isnull().sum()

fraud_counts=full_data['is_fraud'].value_counts()
#sns.barplot(x=fraud_counts.index, y=fraud_counts.values)
#plt.title('Pre-Process Distribution of Fraud')
#plt.xlabel('Fraud')
#plt.ylabel('Count')
#plt.show()

full_data.duplicated().sum()
full_data.info()

# Data Processing
fraud=full_data[full_data["is_fraud"]==1]
not_fraud=full_data[full_data["is_fraud"]==0]
print(fraud.shape[0])
print(not_fraud.shape[0])

not_fraud=not_fraud.sample(fraud.shape[0])
data=pd.concat([fraud,not_fraud])

fraud_counts=data['is_fraud'].value_counts()
#sns.barplot(x=fraud_counts.index,y=fraud_counts.values)
#plt.title('Post-Process Distribution of Fraud')
#plt.xlabel('Fraud')
#plt.ylabel('Count')
#plt.show()

# drop unnecessary columns from dataset
unused_cols=['Unnamed: 0','first','last','unix_time','street','gender','job','dob','city','state','trans_num','merchant']
data.drop(columns=unused_cols,inplace=True)

data.info()

data['trans_date_trans_time']=pd.to_datetime(data['trans_date_trans_time'])
data['trans_day']=data['trans_date_trans_time'].dt.day
data['trans_month']=data['trans_date_trans_time'].dt.month
data['trans_year']=data['trans_date_trans_time'].dt.year
data['trans_hour']=data['trans_date_trans_time'].dt.hour
data['trans_minute']=data['trans_date_trans_time'].dt.minute
data.drop(columns=['trans_date_trans_time'],inplace=True)

encoder=LabelEncoder()
data['category']=encoder.fit_transform(data['category'])
data['cc_num']=encoder.fit_transform(data['cc_num'])

data.head()

# Feature Scaling
scaler=StandardScaler()
data['amt']=scaler.fit_transform(data[['amt']])
data['zip']=scaler.fit_transform(data[['zip']])
data['city_pop']=scaler.fit_transform(data[['city_pop']])
data['cc_num']=encoder.fit_transform(data['cc_num'])

X=data.drop('is_fraud',axis=1)
y=data['is_fraud']

# Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=125)

# Train the Random Forest Classifier
RFC=RandomForestClassifier(random_state=0)
RFC.fit(X_train,y_train)
y_pred=RFC.predict(X_test)

# Evaluate the Model
print('Accuracy:',accuracy_score(y_test, y_pred))
print('F1 score:',f1_score(y_test, y_pred))
print('Confusion matrix:',confusion_matrix(y_test, y_pred))

labels = ["Not Fraud", "Fraud"]
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()
plt.title('RFC Confusion Matrix')
plt.show()

print("Random Forest Classifier model accuracy(in %):", metrics.accuracy_score(y_test, y_pred) * 100)