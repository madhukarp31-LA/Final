import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt


# Load the data
data = pd.read_excel('C:/Users/madhu/OneDrive/Desktop/Filtered_Breaches_Theft.xlsx')

# Example data setup (replace with actual data loading)
# Assuming a simple data structure similar to your context
data = {
    'Year': np.arange(2006, 2024),
    'Total_Individuals': np.random.poisson(lam=200000, size=18)  # Simulated data
}
df = pd.DataFrame(data)

# Assuming the year and the number of individuals affected
X = df[['Year']]  # Features
y = df['Total_Individuals']  # Target

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predict using the linear model
linear_predictions = linear_model.predict(X_test)

# Evaluate the linear model
linear_mse = mean_squared_error(y_test, linear_predictions)
linear_r2 = r2_score(y_test, linear_predictions)

# Prepare input for the year 2024 prediction with linear model
predict_input = pd.DataFrame([[2024]], columns=['Year'])
predicted_2024_linear = linear_model.predict(predict_input)

# Output linear model results
print("Linear Regression Mean Squared Error:", linear_mse)
print("Linear Regression R^2 Score:", linear_r2)
print("Linear Regression Predicted number of individuals affected by theft in 2024:", predicted_2024_linear[0])

# Create a polynomial regression model
degree = 2
poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

# Fit the polynomial model
poly_model.fit(X_train, y_train)

# Predict using the polynomial model
poly_predictions = poly_model.predict(X_test)

# Evaluate the polynomial model
poly_mse = mean_squared_error(y_test, poly_predictions)
poly_r2 = r2_score(y_test, poly_predictions)

# Prepare input for the year 2024 prediction with polynomial model
poly_predicted_2024 = poly_model.predict(predict_input)

# Output polynomial model results
print("Polynomial Regression Mean Squared Error:", poly_mse)
print("Polynomial Regression R^2 Score:", poly_r2)
print("Polynomial Regression Predicted number of individuals affected by theft in 2024:", poly_predicted_2024[0])

# Plotting the results for visualization
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, linear_model.predict(X), color='red', label='Linear Model')
plt.plot(X, poly_model.named_steps['linearregression'].predict(PolynomialFeatures(degree).fit_transform(X)), color='green', label='Polynomial Model')
plt.title('Regression Analysis')
plt.xlabel('Year')
plt.ylabel('Total Individuals')
plt.legend()
plt.show()
