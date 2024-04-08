import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the data
data = pd.read_excel(r'C:\Users\hartzellar22\Downloads\Restaurant Revenue.xlsx')

# Select the features and target variable
X = data[['Number_of_Customers', 'Menu_Price', 'Marketing_Spend', 'Average_Customer_Spending', 'Promotions', 'Reviews']]
y = data['Monthly_Revenue']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Create a new restaurant with randomly generated variables
new_restaurant = pd.DataFrame({
    'Number_of_Customers': np.random.randint(50, 200),
    'Menu_Price': np.random.randint(10, 50),
    'Marketing_Spend': np.random.randint(100, 1000),
    'Average_Customer_Spending': np.random.randint(20, 100),
    'Promotions': np.random.randint(0, 5),
    'Reviews': np.random.randint(0, 100)
}, index=[0])

# Predict the monthly revenue for the new restaurant
new_revenue = model.predict(new_restaurant)
print(f"Predicted Monthly Revenue for New Restaurant: ${new_revenue[0]:.2f}")
