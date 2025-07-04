import pandas as pd

train_df = pd.read_csv("churn_data-training.csv")
test_df = pd.read_csv("churn_data-testing.csv")
df = pd.concat([train_df, test_df], ignore_index=True)

print("Train Shape:", train_df.shape)
print("Test Shape:", test_df.shape)

print(train_df.columns)
print(train_df.info())
print(train_df.isnull().sum())

print(train_df.head())

print(train_df['Churn'].value_counts())

print(train_df.describe())

import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Churn', data=train_df)
plt.title("Churn Distribution")
# plt.show()

sns.countplot(x='Gender', hue='Churn', data=train_df)
plt.title("Gender-wise Churn")
# plt.show()


# churn count plot
import seaborn as sns
import matplotlib.pyplot as plt

# gender VS churn
sns.countplot(x='Gender', hue='Churn', data=train_df)
plt.title("gender-wise churn")
# plt.show()

# contract type VS churn
sns.countplot(x='Contract Length', hue='Churn', data=train_df)
plt.title("contract type VS churn")
plt.xticks(rotation=45)
# plt.show()

# monthly charges distribution
sns.histplot(data=train_df, x='Payment Delay', hue='Churn', kde=True)
plt.title("monthly charges distribution")
# plt.show()


print(train_df.describe())
print( train_df.dtypes)

# encoding methon
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# split x and y
x = df.drop('churn', axis=1)
y = df['churn']

# feature scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, train_size=0.2, random_state=42)

# Train Multiple Models
# Train 3 algorithms: Logistic Regression, Decision Tree, Random Forest

from sklearn.linear_model import LogisticRegression
lm = LogisticRegression()
lm.fit(x_train, y_train)
y_pred_lm = lm.predict(x_test)

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)
y_pred_tree = tree.predict(x_test)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
y_pred_rfc = rfc.predict(x_test)

from sklearn.metrics import accuracy_score

print("logistic accuracy:", accuracy_score(y_pred_lm, y_test))
print("decision tree accuracy:", accuracy_score(y_pred_tree, y_test))
print("random forest:", accuracy_score(y_pred_rfc, y_test))

import joblib
joblib.dump(rfc, 'churn_model.pkl') # use the best performaing model here

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
x['gender'] = le.fit_transform(x['gender'])  # 'Female' → 0, 'Male' → 1





