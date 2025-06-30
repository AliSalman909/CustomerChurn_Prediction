import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv(r"C:\Users\Ali's HP\Desktop\Internship\task3\Churn_Modelling.csv")
print("Data Shape:", df.shape)
print(df.head())

# Dropping the columns which arent in need
# row number is simply an index, cusotmer id is a primary key and surname is not required 
df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)  

# Encode Categorical Columns
df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)
# New columns created like Geography_Germany, Geography_Spain, Gender_Male etc so models can understand them

X = df.drop('Exited', axis=1)  # x has all the columns other than the last one 'exited'
y = df['Exited']               # y has the exited column

#2 sets will be made, one ofr train and one for test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

model = RandomForestClassifier(n_estimators=100, random_state=50)  #creating model using RandomForest algorithim. 100 trees with 42 randomness
#train random forest with trainign data
model.fit(X_train, y_train)  
y_pred = model.predict(X_test) #1 or 0 regarding customer churn


# Evaluate the Model. y_test is to check whether customer exited and y_pred is what model predicted
acc = accuracy_score(y_test, y_pred)
print("Model Accuracy-> ", acc)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# Analyze Feature Importance
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh', figsize=(10, 6), color='teal')
plt.title("Feature Importance in Predicting Churn")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()
