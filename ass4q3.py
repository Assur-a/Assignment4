import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# Area of houses
X = np.array([
    500, 700, 900, 1100, 1300,
    1500, 1700, 1900, 2100, 2300,
    2500, 2700, 2900, 3100
]).reshape(-1, 1)

# Non-linear house prices
y = np.array([
    25, 32, 40, 48, 58,
    68, 79, 91, 104, 118,
    133, 149, 166, 184
])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# -----------------------------
# Linear Regression
# -----------------------------

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)

linear_r2 = r2_score(y_test, linear_pred)

# -----------------------------
# Polynomial Regression
# -----------------------------

polynomial_model = Pipeline([
    ("polynomial_features", PolynomialFeatures(degree=2)),
    ("linear_regression", LinearRegression())
])

polynomial_model.fit(X_train, y_train)

poly_pred = polynomial_model.predict(X_test)

poly_r2 = r2_score(y_test, poly_pred)

# Results
print("--- Model Comparison ---")
print(f"Linear Regression R2    : {linear_r2:.4f}")
print(f"Polynomial Regression R2 : {poly_r2:.4f}")

# Plot
X_plot = np.linspace(
    X.min(),
    X.max(),
    200
).reshape(-1, 1)

plt.figure(figsize=(8, 5))

plt.scatter(
    X,
    y,
    label="Actual Data"
)

plt.plot(
    X_plot,
    linear_model.predict(X_plot),
    linewidth=2,
    label="Linear Regression"
)

plt.plot(
    X_plot,
    polynomial_model.predict(X_plot),
    linewidth=2,
    label="Polynomial Regression"
)

plt.xlabel("House Area (sq. ft.)")
plt.ylabel("House Price")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.grid(True)

plt.show()