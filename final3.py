import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

# Load the dataset
data = pd.read_csv('cybersecurity_attacks.csv')
print("Data loaded successfully.")


# Convert categorical data to numeric if necessary
severity_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
data['Severity Level'] = data['Severity Level'].map(severity_mapping)

# Selecting relevant numerical columns
numerical_data = data[['Packet Length', 'Anomaly Scores', 'Severity Level']]

# Standardizing the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_data)

# Initialize anomaly detection models
iso_forest = IsolationForest(contamination=0.05)
oc_svm = OneClassSVM(nu=0.05)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)

# Fit the models
iso_forest_pred = iso_forest.fit_predict(scaled_data)
oc_svm_pred = oc_svm.fit_predict(scaled_data)
lof_pred = lof.fit_predict(scaled_data)

# Calculate the number of anomalies detected by each model
iso_forest_anomalies = (iso_forest_pred == -1).sum()
oc_svm_anomalies = (oc_svm_pred == -1).sum()
lof_anomalies = (lof_pred == -1).sum()

# Output the results
print(f'Isolation Forest detected {iso_forest_anomalies} anomalies')
print(f'One-Class SVM detected {oc_svm_anomalies} anomalies')
print(f'Local Outlier Factor detected {lof_anomalies} anomalies')
