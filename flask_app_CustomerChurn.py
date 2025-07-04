from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)

import pandas as pd

train_df = pd.read_csv("churn_data-training.csv")
test_df = pd.read_csv("churn_data-testing.csv")
df = pd.concat([train_df, test_df], ignore_index=True)

df.rename(columns={"Churn": "churn"}, inplace=True)
df.dropna(subset=['churn'], inplace=True)
# df = train_df.copy()

x = df.drop('churn', axis=1)
y = df['churn']

le = LabelEncoder()
x['Gender'] = le.fit_transform(x['Gender'])  # 'Female' → 0, 'Male' → 1

x = pd.get_dummies(x, drop_first=True)
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

from sklearn.ensemble import RandomForestClassifier
import joblib

model = RandomForestClassifier()
model.fit(x_scaled, y)  # Make sure x_scaled and y are defined

joblib.dump(model, 'churn_model.pkl')
# Then load in app
model = joblib.load('churn_model.pkl')

joblib.dump(scaler, 'scaler.pkl')
scaler = joblib.load('scaler.pkl')

x = pd.get_dummies(x, drop_first=True)
joblib.dump(x.columns.tolist(), 'model_columns.pkl')
model_columns = joblib.load('model_columns.pkl')



# dummy lableencoder (you'll need the same logic you used in training)
# le = LabelEncoder()
# scaler = StandardScaler()

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        Gender = request.form['Gender']
        Payment_delay = float(request.form['Payment Delay'])
        age = float(request.form['Age'])
        tenure = float(request.form['Tenure'])
        usage = float(request.form['Usage frequency'])
        support_calls = float(request.form['Support calls'])
        sub_type = float(request.form['Subscription Type'])
        contract_length = float(request.form['Contract Length'])
        total_spend = float(request.form['Total Spend'])
        last_interaction = float(request.form['Last Interaction'])

        Gender = 1 if Gender == 'Male' else 0

        data = pd.DataFrame([[Gender, Payment_delay, age, tenure, usage,
                              support_calls, sub_type, contract_length, total_spend,
                              last_interaction]],
                              columns = ['Gender', 'Payment Delay', 'Age', 'Tenure', 'Usage frequency',
                                         'Support calls', 'Subscription Type', 'Contract Length', 'Total Spend',
                                         'Last Interaction'])
        
        # 3. Apply one-hot encoding like training
        data = pd.get_dummies(data)

        # 4. Reindex to match model_columns
        data = data.reindex(columns=model_columns, fill_value=0)

        scaler = joblib.load('scaler.pkl')
        scaled_data = scaler.transform(data)


        prediction = model.predict(scaled_data)
        result = "churn" if prediction[0] == 1 else "not churn"

        return render_template("result.html", prediction=result)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
