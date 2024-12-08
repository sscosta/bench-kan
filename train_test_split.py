from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import pandas as pd

# Load data
ds = pd.read_csv("preprocessed_data/har_mobile.csv")

# Separate features and label

# Encode categorical columns (assuming 'subject' is also categorical)
categorical_cols = ['subject', 'activity']
le = LabelEncoder()
for col in categorical_cols:
  ds[col] = le.fit_transform(ds[col])


label = ds['activity']
features = ds.drop(columns=['activity'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.3, random_state=42)

# Combine features and labels with renamed label column
train_data = pd.concat([X_train, y_train.rename('activity')], axis=1)
test_data = pd.concat([X_test, y_test.rename('activity')], axis=1)

# Save data to CSV
train_data.to_csv("preprocessed_data/har_mobile_train.csv", index=False)
test_data.to_csv("preprocessed_data/har_mobile_test.csv", index=False)

print("Training and testing data saved to CSV files with encoded categorical columns and labels.")
