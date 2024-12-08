import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
df = pd.read_csv('final_combined_output.csv')

df['timestamp'] = pd.to_datetime(df['timestamp'])
df.isnull().sum()
df = df.dropna()  # Simple approach: drop rows with missing values


scaler = StandardScaler()
df[['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']] = scaler.fit_transform(
    df[['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']]
)
def segment_signal(data, window_size):
    segments = []
    labels = []
    for i in range(0, len(data) - window_size, window_size):
        accel_x = data['accel_x'].values[i: i + window_size]
        accel_y = data['accel_y'].values[i: i + window_size]
        accel_z = data['accel_z'].values[i: i + window_size]
        gyro_x = data['gyro_x'].values[i: i + window_size]
        gyro_y = data['gyro_y'].values[i: i + window_size]
        gyro_z = data['gyro_z'].values[i: i + window_size]
        label = data['activity'][i: i + window_size].mode()[0]
        segments.append([accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z])
        labels.append(label)
    return segments, labels

window_size = 100
segments, labels = segment_signal(df, window_size)



def extract_features(segment):
    features = []
    for i in range(len(segment)):
        features.append(np.mean(segment[i]))
        features.append(np.std(segment[i]))
        features.append(np.min(segment[i]))
        features.append(np.max(segment[i]))
    return features

X = np.asarray([extract_features(segment) for segment in segments])
y = np.asarray(labels)

le = LabelEncoder()
y_encoded = le.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

train_data = pd.DataFrame(X_train)
train_data['label'] = y_train
test_data = pd.DataFrame(X_test)
test_data['label'] = y_test

train_data.to_csv('har_mobile_train.csv', index=False)
test_data.to_csv('har_mobile_test.csv', index=False)

#clf = RandomForestClassifier(n_estimators=100, random_state=42)
#clf.fit(X_train, y_train)


#y_pred = clf.predict(X_test)
#print("Accuracy:", accuracy_score(y_test, y_pred))
#print(classification_report(y_test, y_pred, target_names=le.classes_))
