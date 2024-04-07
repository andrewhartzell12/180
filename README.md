import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the data from the spreadsheet
data = pd.read_excel('C:/Users/andre/Downloads/baseball.xlsx')

# Select the relevant features and target variable
features = ['Runs Scored', 'Runs Allowed', 'Wins', 'OBP', 'SLG', 'Team Batting Average']
target = 'Playoffs'

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Use the model to predict playoffs for a team with randomly generated statistics
new_team_stats = pd.DataFrame([[800, 700, 90, 0.350, 0.450, 0.280]], columns=features)
prediction = model.predict(new_team_stats)
print(f"Prediction: {'Yes' if prediction[0] == 1 else 'No'}")
