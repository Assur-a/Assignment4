import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Features:
# Area in sq.ft.
# Bedrooms
# Age of house in years

X = np.array([
    [800, 2, 15],
    [1000, 2, 10],
    [1200, 3, 8],
    [1400, 3, 6],
    [1600, 3, 5],
    [1800, 4, 4],
    [2000, 4, 3],
    [2200, 4, 2],
    [2400, 5, 2],
    [2600, 5, 1],
    [2800, 5, 1],
    [3000, 6, 1]
])

# House price in lakhs
y = np.array([
    35, 42, 48, 55, 63, 72,
    80, 88, 96, 105, 114, 125
])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# Model
model = LinearRegression()

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("--- Multiple Linear Regression ---")

print("\nCoefficients:")
print("Area:", model.coef_[0])
print("Bedrooms:", model.coef_[1])
print("Age:", model.coef_[2])

print("\nIntercept:", model.intercept_)

print("\n--- Evaluation Metrics ---")
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}")

# Predict a new house
new_house = np.array([[2000, 4, 3]])

predicted_price = model.predict(new_house)

print(
    "\nPredicted price for 2000 sq.ft., "
    "4 bedrooms, 3 years old house:",
    f"{predicted_price[0]:.2f} lakhs"
)