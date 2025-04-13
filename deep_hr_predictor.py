
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('./HR_comma_sep.csv')

# Preprocessing

# Target encoding for categorical features
label_columns = ['salary', 'sales']  # Assuming target encoding for these
for col in label_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# Feature scaling (consider if features have different scales)
scaler = StandardScaler()

# Define X_columns based on your actual features
X_columns = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours',
              'time_spend_company', 'Work_accident', 'promotion_last_5years']  # Update based on your features

# Scale the features
data[X_columns] = scaler.fit_transform(data[X_columns])

# Splitting data into features and target
X = data[X_columns]  # Select features based on X_columns
y = data['left']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Building the TensorFlow/Keras model with hyperparameter tuning
model = Sequential([
    Dense(units=64, activation='relu', input_shape=(X_train.shape[1],)),  # First hidden layer
    Dropout(0.2),  # Dropout for regularization
    Dense(units=32, activation='relu'),  # Second hidden layer
    Dropout(0.2),
    Dense(units=1, activation='sigmoid')  # Output layer for binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model training (consider early stopping and other techniques)
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model performance
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Plot training and validation accuracy/loss
plt.figure(figsize=(10, 6))

# Accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.grid(True)

# Loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss (Binary Crossentropy)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.grid(True)

# Improve visibility (optional)
plt.tight_layout()

plt.show()