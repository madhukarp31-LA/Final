import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('cybersecurity_attacks.csv')
print("Data loaded successfully.")


# Selecting features and the target
features = data[['Source Port', 'Destination Port', 'Protocol', 'Packet Length', 'Packet Type', 'Traffic Type']].copy()
target = data['Attack Type']

# Encoding categorical variables
for col in features.columns:
    if features[col].dtype == 'object':  # Only apply to object type columns
        le = LabelEncoder()
        features.loc[:, col] = le.fit_transform(features[col])  # Use .loc to ensure modifications are done in place

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=10000, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training the RandomForest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

# Making predictions on the test set
y_pred = clf.predict(X_test_scaled)

# Calculating the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2%}')
