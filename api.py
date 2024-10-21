from fastapi import FastApi
import pickle
import pandas as pd

app = FastApi()

# Load voting classifier
with open('voting_clf_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Preprocess customer data

def preprocess_data(customer_dict):
    input_dict = {
        "CreditScore": customer_dict['CreditScore'],
        "Age": customer_dict['Age'],
        "Tenure": customer_dict['Tenure'],
        "Balance": customer_dict['Balance'],
        "NumOfProducts": customer_dict['NumOfProducts'],
        "HasCrCard": customer_dict['HasCrCard'],
        "IsActiveMember": customer_dict['IsActiveMember'],
        "EstimatedSalary": customer_dict['EstimatedSalary'],
        "Geography_France": customer_dict['location'] == "France",
        "Geography_Germany": customer_dict['location'] == "Germany",
        "Geography_Spain": customer_dict['location'] == "Spain",
        "Gender_Female": customer_dict['gender'] == "Female",
        "Gender_Male": customer_dict['gender'] == "Male",
    }
    customer_df = pd.DataFrame[input_dict]

    return customer_df

# Functions to make predictions
def get_predictions(customer_dict):
    preprocessed_data = preprocess_data(customer_dict)
    prediction = loaded_model.predict(preprocessed_data)
    probability = loaded_model.predict_proba(preprocessed_data)

    return prediction, probability

@app.post('/predict')
async def predcit(data: dict):
    # Make prediction
    prediction, probability = get_predictions(data)

    return {
        'prediction': prediction.tolist(),
        'probability': probability.tolist(),
    }

if __name__ == '__api__' :
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=2525)
