import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# Read Dataset
data = pd.read_csv('housing.csv')

# Features
features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
target = 'price'

X = data[features]
y = data[target]

# Train/Dev/Test
total_samples = len(X)
train_samples = 250
dev_samples = 50
test_samples = 45

X_train = X[:train_samples]
y_train = y[:train_samples]

X_dev = X[train_samples:train_samples + dev_samples]
y_dev = y[train_samples:train_samples + dev_samples]

X_test = X[-test_samples:]
y_test = y[-test_samples:]

# StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_dev = scaler.transform(X_dev)
X_test = scaler.transform(X_test)

# Using Keras
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_dim=5),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Output layer
])

model.compile(optimizer='adam', loss='mean_squared_error')

# train model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_dev, y_dev))

# Evaluation
loss = model.evaluate(X_test, y_test)
print(f"Mean Squared Error on Test Data: {loss}")

# Predict
predictions = model.predict(X_test)

# Result
for i in range(len(predictions)):
    print(f"Giá trị dự đoán: {predictions[i][0]}, Giá trị thực tế: {y_test.iloc[i]}")

