import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# House area in square feet
X = np.array([
    800, 1000, 1200, 1400, 1600,
    1800, 2000, 2200, 2400, 2600,
    2800, 3000
]).reshape(-1, 1)

# House prices in lakhs
y = np.array([
    35, 42, 48, 55, 62,
    70, 78, 85, 93, 101,
    110, 120
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# Create model
model = LinearRegression()

# Train
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("--- Simple Linear Regression ---")
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

print("\n--- Evaluation Metrics ---")
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}")

# Plot
plt.figure(figsize=(8, 5))

plt.scatter(
    X,
    y,
    label="Actual Data"
)

plt.plot(
    X,
    model.predict(X),
    linewidth=2,
    label="Regression Line"
)

plt.xlabel("House Area (sq. ft.)")
plt.ylabel("House Price (Lakhs)")
plt.title("House Area vs House Price")
plt.legend()
plt.grid(True)

plt.show()