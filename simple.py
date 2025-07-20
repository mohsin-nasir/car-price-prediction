import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

data = pd.read_csv('price_data.csv')
data = data.dropna()

X = data[['year', 'km_driven', 'mileage(km/ltr/kg)', 'seats']]
y = data['selling_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train, y_train)

def predict_price():
    year = int(input("Enter year: "))
    km_driven = int(input("Enter km driven: "))
    mileage = float(input("Enter mileage: "))
    seats = int(input("Enter seats: "))

    input_data = [[year, km_driven, mileage, seats]]
    # input_data = pd.DataFrame([[year, km_driven, mileage, seats]],columns=X.columns) // To simplify the code the dataframe part has been commented
    price = model.predict(input_data)[0]
    print(price)

predict_price()