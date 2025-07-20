import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

data = pd.read_csv('price_data.csv')
data = data.dropna()

X = data[['year', 'km_driven', 'fuel', 'owner', 'mileage(km/ltr/kg)', 'seats']]
y = data['selling_price']

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train, y_train)

def predict_price():
    year = int(input("Enter year: "))
    km_driven = int(input("Enter km driven: "))
    fuel = input("Enter fuel: ")
    owner = input("Enter owner: ")
    mileage = float(input("Enter mileage: "))
    seats = int(input("Enter seats: "))

    input_df = pd.DataFrame([{
        'year': year,
        'km_driven': km_driven,
        'fuel': fuel,
        'owner': owner,
        'mileage(km/ltr/kg)': mileage,
        'seats': seats
    }])

    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    price = model.predict(input_df)[0]
    print("Predicted selling price:", price)

predict_price()

