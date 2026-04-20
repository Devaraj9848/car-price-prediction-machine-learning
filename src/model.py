import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib

# ===============================
# Load Dataset
# ===============================
data = pd.read_csv('../data/car_data.csv')

print("Dataset Preview:")
print(data.head())

# ===============================
# Preprocessing
# ===============================
data.dropna(inplace=True)

# Convert categorical to numeric
data = pd.get_dummies(data, drop_first=True)

X = data.drop('Selling_Price', axis=1)
y = data['Selling_Price']

# ===============================
# Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# Scaling (for SVM)
# ===============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# Evaluation Function
# ===============================
def evaluate(name, model, X_t, y_t):
    pred = model.predict(X_t)
    print(f"\n{name}")
    print("MAE:", mean_absolute_error(y_t, pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_t, pred)))
    print("R2:", r2_score(y_t, pred))

# ===============================
# Models
# ===============================

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
evaluate("Linear Regression", lr, X_test, y_test)

# Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
evaluate("Lasso", lasso, X_test, y_test)

# Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
evaluate("Ridge", ridge, X_test, y_test)

# ElasticNet
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic.fit(X_train, y_train)
evaluate("ElasticNet", elastic, X_test, y_test)

# Decision Tree
dt = DecisionTreeRegressor(max_depth=5)
dt.fit(X_train, y_train)
evaluate("Decision Tree", dt, X_test, y_test)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
evaluate("Random Forest", rf, X_test, y_test)

# SVM
svm = SVR(kernel='rbf')
svm.fit(X_train_scaled, y_train)
evaluate("SVM", svm, X_test_scaled, y_test)

# ===============================
# Hyperparameter Tuning (Random Forest)
# ===============================
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 5, 10]
}

grid = GridSearchCV(RandomForestRegressor(), param_grid, cv=5)
grid.fit(X_train, y_train)

print("\nBest Parameters:", grid.best_params_)

best_model = grid.best_estimator_
evaluate("Tuned Random Forest", best_model, X_test, y_test)

# ===============================
# Save Model
# ===============================
joblib.dump(best_model, '../models/car_price_model.pkl')
joblib.dump(scaler, '../models/scaler.pkl')

print("\nModel saved successfully!")
