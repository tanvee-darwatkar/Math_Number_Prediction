import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load the training and testing datasets
train_data = pd.read_csv('train.csv')  # training data
test_data = pd.read_csv('test.csv')    # testing data

print("Data Loaded Successfully\n")

#  Basic Exploration of the Training Data
print("First 5 rows of the training dataset:")
print(train_data.head())

print("\nInformation about columns and missing values:")
print(train_data.info())

print("\nBasic statistics for numerical columns:")
print(train_data.describe())

print("\nChecking for missing values in each column:")
print(train_data.isnull().sum())

# Encode categorical columns using One-Hot Encoding
categorical_cols = [
    'gender',
    'race/ethnicity',
    'parental level of education',
    'lunch',
    'test preparation course'
]

# Convert categorical columns to numeric dummy variables
train_encoded = pd.get_dummies(train_data, columns=categorical_cols, drop_first=True)
test_encoded = pd.get_dummies(test_data, columns=categorical_cols, drop_first=True)

# Align the test set with train columns (fill any missing columns with 0)
test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)

print("\n One-Hot Encoding complete. Columns converted to numeric form.")
print(train_encoded.head())

#Separate Features and Target Variable
X_train = train_encoded.drop('math score', axis=1)
y_train = train_encoded['math score']

X_test = test_encoded.drop('math score', axis=1)
y_test = test_encoded['math score']

print("\nFeatures (first 5 rows):")
print(X_train.head())

print("\nTarget variable (math scores):")
print(y_train.head())

#  Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)
print("\n Model training complete.")

# Make Predictions on Test Data
y_pred = model.predict(X_test)
print("\n Prediction on test data complete.")

#  Evaluate Model Performance
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n Model Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f} → Average difference between predicted & actual scores")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} → Typical size of prediction errors")
print(f"R-squared (R²) Score: {r2:.2f} → Model’s ability to explain score variation")

#  Display comparison of actual vs predicted
comparison = pd.DataFrame({
    'Actual Math Score': y_test.values,
    'Predicted Math Score': y_pred
})
print("\nSample comparison of Actual vs Predicted values:")
print(comparison.head())
