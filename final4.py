
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('cybersecurity_attacks.csv')
print("Data loaded successfully.")


# Convert 'Timestamp' from object to datetime and extract useful features
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data['Hour'] = data['Timestamp'].dt.hour
data['Minute'] = data['Timestamp'].dt.minute
data['Day'] = data['Timestamp'].dt.day
data['Month'] = data['Timestamp'].dt.month
data['Year'] = data['Timestamp'].dt.year
data.drop('Timestamp', axis=1, inplace=True)

# Handling IP addresses - simplistic approach: split into octets and convert to integers
data['Source IP Address'] = data['Source IP Address'].apply(lambda x: int(x.split('.')[0]))
data['Destination IP Address'] = data['Destination IP Address'].apply(lambda x: int(x.split('.')[0]))

# Encode categorical variables using Label Encoding
categorical_cols = data.select_dtypes(include=['object']).columns  # Automatically select non-numeric columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # Store label encoder for each column

# Define features and target variable
X = data.drop('Attack Type', axis=1)  # Assuming 'Attack Type' is the target variable
y = data['Attack Type']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Initialize and train a Random Forest classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Predict on the test set and calculate accuracy
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Output the model accuracy
print(f'Model accuracy: {accuracy * 100:.2f}%')
