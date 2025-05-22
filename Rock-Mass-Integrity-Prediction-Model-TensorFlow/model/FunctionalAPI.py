import pandas as pd
import numpy as np
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

# Load text and image data
text_file_path = ''
image_file_path = ''

# Read text features
text_df = pd.read_excel(text_file_path, header=1)
# Read image features
image_df = pd.read_excel(image_file_path, header=0)

# Data cleaning: convert to numeric and drop rows with missing values
text_numeric_cols = text_df.columns[2:]
text_df[text_numeric_cols] = text_df[text_numeric_cols].apply(pd.to_numeric, errors='coerce')
text_df = text_df.dropna()

image_numeric_cols = image_df.columns[2:]
image_df[image_numeric_cols] = image_df[image_numeric_cols].apply(pd.to_numeric, errors='coerce')
image_df = image_df.dropna()

# Extract features and labels
X_text = text_df.iloc[:, 2:].values
# Convert labels from [1,5] to [0,4]
y_text = text_df.iloc[:, 1].values - 1

X_image = image_df.iloc[:, 2:].values
# Convert labels from [1,5] to [0,4]
y_image = image_df.iloc[:, 1].values - 1

# Split into training and test sets
X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(X_text, y_text, test_size=0.2, random_state=42)
X_train_image, X_test_image, y_train_image, y_test_image = train_test_split(X_image, y_image, test_size=0.2,
                                                                            random_state=42)

# Align the sample sizes
min_train_samples = min(X_train_text.shape[0], X_train_image.shape[0])
min_test_samples = min(X_test_text.shape[0], X_test_image.shape[0])

X_train_text = X_train_text[:min_train_samples]
X_train_image = X_train_image[:min_train_samples]
y_train = y_train_text[:min_train_samples]

X_test_text = X_test_text[:min_test_samples]
X_test_image = X_test_image[:min_test_samples]
y_test = y_test_text[:min_test_samples]

# One-hot encode labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Feature scaling
scaler_text = StandardScaler()
X_train_text = scaler_text.fit_transform(X_train_text)
X_test_text = scaler_text.transform(X_test_text)

scaler_image = StandardScaler()
X_train_image = scaler_image.fit_transform(X_train_image)
X_test_image = scaler_image.transform(X_test_image)

# Build multi-input model
input_text = Input(shape=(X_train_text.shape[1],), name='text_input')
x_text = Dense(64, activation='relu')(input_text)
x_text = Dense(32, activation='relu')(x_text)

input_image = Input(shape=(X_train_image.shape[1],), name='image_input')
x_image = Dense(64, activation='relu')(input_image)
x_image = Dense(32, activation='relu')(x_image)

combined = Concatenate()([x_text, x_image])
x = Dense(64, activation='relu')(combined)
x = Dense(32, activation='relu')(x)

output = Dense(5, activation='softmax')(x)

model = Model(inputs=[input_text, input_image], outputs=output)

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
start_train_time = time.time()
history = model.fit([X_train_text, X_train_image], y_train, epochs=50, batch_size=32, validation_split=0.1)
end_train_time = time.time()
train_time = end_train_time - start_train_time
print(f"Training time: {train_time:.2f} seconds")

# Plot accuracy curve
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

# Plot loss curve
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

# Save training history
history_df = pd.DataFrame({
    'Epoch': np.arange(1, len(history.history['accuracy']) + 1),
    'Training Accuracy': history.history['accuracy'],
    'Validation Accuracy': history.history['val_accuracy'],
    'Training Loss': history.history['loss'],
    'Validation Loss': history.history['val_loss']
})
history_df.to_excel('combined_training_history.xlsx', index=False)

# Test set prediction
start_test_time = time.time()
y_pred = model.predict([X_test_text, X_test_image])
end_test_time = time.time()
test_time = end_test_time - start_test_time
print(f"Prediction time: {test_time:.2f} seconds")

y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Print classification report
print(classification_report(y_true_classes, y_pred_classes))

baseline_accuracy = accuracy_score(y_true_classes, y_pred_classes)
print(f"Baseline Accuracy: {baseline_accuracy:.2f}")


# Feature importance: permutation for text features
def permutation_importance_text(model, X_test_text, X_test_image, y_test, baseline_accuracy):
    feature_importance = []
    for i in range(X_test_text.shape[1]):
        X_test_text_permuted = X_test_text.copy()
        np.random.shuffle(X_test_text_permuted[:, i])
        y_pred_permuted = model.predict([X_test_text_permuted, X_test_image])
        y_pred_permuted_classes = np.argmax(y_pred_permuted, axis=1)
        permuted_accuracy = accuracy_score(y_true_classes, y_pred_permuted_classes)
        importance = baseline_accuracy - permuted_accuracy
        feature_importance.append(importance)
    return feature_importance


# Feature importance: permutation for image features
def permutation_importance_image(model, X_test_text, X_test_image, y_test, baseline_accuracy):
    feature_importance = []
    for i in range(X_test_image.shape[1]):
        X_test_image_permuted = X_test_image.copy()
        np.random.shuffle(X_test_image_permuted[:, i])
        y_pred_permuted = model.predict([X_test_text, X_test_image_permuted])
        y_pred_permuted_classes = np.argmax(y_pred_permuted, axis=1)
        permuted_accuracy = accuracy_score(y_true_classes, y_pred_permuted_classes)
        importance = baseline_accuracy - permuted_accuracy
        feature_importance.append(importance)
    return feature_importance


# Compute feature importances
feature_importance_text = permutation_importance_text(model, X_test_text, X_test_image, y_test, baseline_accuracy)
features_text = text_df.columns[2:]
importance_df_text = pd.DataFrame({'Feature': features_text, 'Importance': feature_importance_text, 'Type': 'Text'})

feature_importance_image = permutation_importance_image(model, X_test_text, X_test_image, y_test, baseline_accuracy)
features_image = image_df.columns[2:]
importance_df_image = pd.DataFrame({'Feature': features_image, 'Importance': feature_importance_image, 'Type': 'Image'})

# Combine and show top 20 features
importance_combined = pd.concat([importance_df_text, importance_df_image], ignore_index=True)
importance_combined = importance_combined.sort_values(by='Importance', ascending=False)
top_20_importance_combined = importance_combined.head(20)
print(top_20_importance_combined)

# Plot top 20 feature importances
plt.figure(figsize=(12, 8))
plt.barh(top_20_importance_combined['Feature'].astype(str), top_20_importance_combined['Importance'], color='skyblue')
plt.title('Top 20 Feature Importance based on Permutation (Combined Text and Image)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.gca().invert_yaxis()
plt.grid(True)
plt.show()

top_20_importance_combined.to_excel('combined_permutation_feature_importance_top20.xlsx', index=False)


# Sensitivity analysis for text features
def sensitivity_analysis_text(model, X_test_text, X_test_image, y_test, perturbation=0.05):
    sensitivity_scores = []
    baseline_accuracy = accuracy_score(y_test, np.argmax(model.predict([X_test_text, X_test_image]), axis=1))
    for i in range(X_test_text.shape[1]):
        X_test_text_perturbed = X_test_text.copy()
        X_test_text_perturbed[:, i] += X_test_text[:, i] * perturbation
        accuracy_perturbed = accuracy_score(
            y_true_classes, np.argmax(model.predict([X_test_text_perturbed, X_test_image]), axis=1))
        positive_sensitivity = abs(baseline_accuracy - accuracy_perturbed)
        X_test_text_perturbed[:, i] -= 2 * X_test_text[:, i] * perturbation
        accuracy_perturbed = accuracy_score(
            y_true_classes, np.argmax(model.predict([X_test_text_perturbed, X_test_image]), axis=1))
        negative_sensitivity = abs(baseline_accuracy - accuracy_perturbed)
        sensitivity_scores.append((positive_sensitivity + negative_sensitivity) / 2)
    return sensitivity_scores


# Sensitivity analysis for image features
def sensitivity_analysis_image(model, X_test_text, X_test_image, y_test, perturbation=0.05):
    sensitivity_scores = []
    baseline_accuracy = accuracy_score(y_test, np.argmax(model.predict([X_test_text, X_test_image]), axis=1))
    for i in range(X_test_image.shape[1]):
        X_test_image_perturbed = X_test_image.copy()
        X_test_image_perturbed[:, i] += X_test_image[:, i] * perturbation
        accuracy_perturbed = accuracy_score(
            y_true_classes, np.argmax(model.predict([X_test_text, X_test_image_perturbed]), axis=1))
        positive_sensitivity = abs(baseline_accuracy - accuracy_perturbed)
        X_test_image_perturbed[:, i] -= 2 * X_test_image[:, i] * perturbation
        accuracy_perturbed = accuracy_score(
            y_true_classes, np.argmax(model.predict([X_test_text, X_test_image_perturbed]), axis=1))
        negative_sensitivity = abs(baseline_accuracy - accuracy_perturbed)
        sensitivity_scores.append((positive_sensitivity + negative_sensitivity) / 2)
    return sensitivity_scores


# Compute and display top 20 sensitivity features
sensitivity_scores_text = sensitivity_analysis_text(model, X_test_text, X_test_image, y_true_classes)
sensitivity_df_text = pd.DataFrame({'Feature': features_text, 'Sensitivity': sensitivity_scores_text, 'Type': 'Text'})
sensitivity_df_text = sensitivity_df_text.sort_values(by='Sensitivity', ascending=False)

sensitivity_scores_image = sensitivity_analysis_image(model, X_test_text, X_test_image, y_true_classes)
sensitivity_df_image = pd.DataFrame(
    {'Feature': features_image, 'Sensitivity': sensitivity_scores_image, 'Type': 'Image'})
sensitivity_df_image = sensitivity_df_image.sort_values(by='Sensitivity', ascending=False)

sensitivity_combined = pd.concat([sensitivity_df_text, sensitivity_df_image], ignore_index=True)
sensitivity_combined = sensitivity_combined.sort_values(by='Sensitivity', ascending=False)

top_20_sensitivity_combined = sensitivity_combined.head(20)
print(top_20_sensitivity_combined)

# Plot top 20 sensitivity scores
plt.figure(figsize=(12, 8))
plt.barh(top_20_sensitivity_combined['Feature'].astype(str), top_20_sensitivity_combined['Sensitivity'], color='salmon')
plt.title('Top 20 Feature Sensitivity based on Perturbation (Combined Text and Image)')
plt.xlabel('Sensitivity')
plt.ylabel('Feature')
plt.gca().invert_yaxis()
plt.grid(True)
plt.show()

top_20_sensitivity_combined.to_excel('combined_sensitivity_analysis_top20.xlsx', index=False)
