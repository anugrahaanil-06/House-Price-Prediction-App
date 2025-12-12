import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

df = pd.DataFrame({
    "bedrooms":      [2, 3, 4, 3, 5, 4, 2, 6],
    "bathrooms":     [1, 2, 3, 2, 4, 3, 1, 5],
    "sqft":          [900, 1400, 1800, 1600, 2400, 2000, 850, 3000],
    "location_score":[7, 8, 9, 6, 10, 8, 5, 9], 
    "price":         [100000, 150000, 210000, 170000, 350000, 260000, 95000, 450000]
})

X = df[["bedrooms", "bathrooms", "sqft", "location_score"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

joblib.dump(model, "house_model.pkl")
joblib.dump(scaler, "house_scaler.pkl")

print("House Price Model and Scaler Saved Successfully!")