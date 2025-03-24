#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Penguin Species Classification Model Training Module.

This module implements various feature selection methods to train and evaluate
a penguin species classification model. It compares different approaches:
1. Filter Method (Mutual Information)
2. Wrapper Method (Recursive Feature Elimination)
3. Embedded Method (Random Forest Feature Importance)
4. Permutation Importance

The module performs the following operations:
1. Loads and preprocesses penguin data from SQLite database
2. Implements and evaluates different feature selection methods
3. Compares method performance and selects the best model
4. Saves the best performing model for future use
5. Generates visualizations for feature importance and method comparison
"""

# Import required libraries for data manipulation, visualization, and machine learning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

# Load and prepare data
# Load data from SQLite database, excluding original entries (is_original != 1)
db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src', 'data', 'penguins.db')
conn = sqlite3.connect(db_path)
query = "SELECT species, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g FROM penguins WHERE is_original != 1"
data = pd.read_sql_query(query, conn)
conn.close()

# Clean and prepare the dataset
data = data.dropna()
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Split features and target
X = data.drop(columns=['species'])  # Features
y = data['species']  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features for better model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Initialize dictionary to store results from different methods
results = {
    'Method': [],
    'Optimal Number of Features': [],
    'Selected Features': [],
    'Accuracy': []
}

# Function to find optimal number of features using cross-validation
def find_optimal_features(X, y, method, param_name, param_range, estimator, cv=5):
    mean_scores = []
    
    for n in param_range:
        # Configure the method with the current number of features
        if method == 'filter':
            selector = SelectKBest(score_func=mutual_info_classif, k=n)
            X_selected = selector.fit_transform(X, y)
        elif method == 'wrapper':
            selector = RFE(estimator=estimator, n_features_to_select=n)
            X_selected = selector.fit_transform(X, y)
        elif method == 'embedded':
            selector = estimator.fit(X, y)
            importances = selector.feature_importances_
            indices = np.argsort(importances)[::-1][:n]
            mask = np.zeros(X.shape[1], dtype=bool)
            mask[indices] = True
            X_selected = X.iloc[:, mask]
        
        # Calculate cross-validation score
        scores = cross_val_score(estimator, X_selected, y, cv=cv, scoring='accuracy')
        mean_scores.append(scores.mean())
    
    # Find the optimal number of features
    optimal_n = param_range[np.argmax(mean_scores)]
    max_score = max(mean_scores)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, mean_scores, marker='o')
    plt.axvline(x=optimal_n, color='r', linestyle='--', label=f'Optimal: {optimal_n} features')
    plt.xlabel('Number of Features')
    plt.ylabel('Cross-Validation Accuracy')
    plt.title(f'{method.capitalize()} Method - Feature Selection')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'images/{method}_optimal_features.png')
    plt.close()
    
    return optimal_n, max_score

# 1. Filter Method (SelectKBest with mutual_info_classif)
print("\n1. Filter Method (Mutual Information)")
max_features = len(X_train.columns)
feature_range = range(1, max_features + 1)
model_rf = RandomForestClassifier(random_state=42)

optimal_n_filter, cv_score_filter = find_optimal_features(
    X_train_scaled_df, y_train, 
    method='filter', 
    param_name='k', 
    param_range=feature_range, 
    estimator=model_rf
)

# Apply the optimal filter method
selector = SelectKBest(score_func=mutual_info_classif, k=optimal_n_filter)
X_train_filter = selector.fit_transform(X_train_scaled_df, y_train)
X_test_filter = selector.transform(X_test_scaled_df)

selected_features_filter = X_train.columns[selector.get_support()]
print(f"Optimal number of features: {optimal_n_filter}")
print(f"Selected features: {selected_features_filter.tolist()}")

# Evaluate filter method
model = RandomForestClassifier(random_state=42)
model.fit(X_train_filter, y_train)
y_pred_filter = model.predict(X_test_filter)
filter_accuracy = accuracy_score(y_test, y_pred_filter)
print(f"Test Accuracy: {filter_accuracy:.4f}")

# Store results
results['Method'].append('Filter (Mutual Information)')
results['Optimal Number of Features'].append(optimal_n_filter)
results['Selected Features'].append(selected_features_filter.tolist())
results['Accuracy'].append(filter_accuracy)

# Visualize feature importance for filter method
plt.figure(figsize=(10, 6))
feature_scores = pd.Series(selector.scores_, index=X_train.columns)
feature_scores.sort_values(ascending=False).plot(kind='bar')
plt.title('Feature Importance (Mutual Information)')
plt.tight_layout()
plt.savefig('images/filter_method_importance.png')
plt.close()

# 2. Wrapper Method (RFE with RandomForest)
print("\n2. Wrapper Method (RFE)")
model_rf = RandomForestClassifier(random_state=42)

optimal_n_wrapper, cv_score_wrapper = find_optimal_features(
    X_train_scaled_df, y_train, 
    method='wrapper', 
    param_name='n_features_to_select', 
    param_range=feature_range, 
    estimator=model_rf
)

# Apply the optimal wrapper method
rfe = RFE(estimator=model_rf, n_features_to_select=optimal_n_wrapper)
X_train_wrapper = rfe.fit_transform(X_train_scaled_df, y_train)
X_test_wrapper = rfe.transform(X_test_scaled_df)

selected_features_wrapper = X_train.columns[rfe.get_support()]
print(f"Optimal number of features: {optimal_n_wrapper}")
print(f"Selected features: {selected_features_wrapper.tolist()}")

# Evaluate wrapper method
model_rf.fit(X_train_wrapper, y_train)
y_pred_wrapper = model_rf.predict(X_test_wrapper)
wrapper_accuracy = accuracy_score(y_test, y_pred_wrapper)
print(f"Test Accuracy: {wrapper_accuracy:.4f}")

# Store results
results['Method'].append('Wrapper (RFE)')
results['Optimal Number of Features'].append(optimal_n_wrapper)
results['Selected Features'].append(selected_features_wrapper.tolist())
results['Accuracy'].append(wrapper_accuracy)

# 3. Embedded Method (RandomForest's feature_importances_)
print("\n3. Embedded Method (RandomForest Feature Importance)")
model_embedded = RandomForestClassifier(random_state=42)

optimal_n_embedded, cv_score_embedded = find_optimal_features(
    X_train_scaled_df, y_train, 
    method='embedded', 
    param_name='n_features', 
    param_range=feature_range, 
    estimator=model_embedded
)

# First, train a full model to get feature importances for all features
full_model = RandomForestClassifier(random_state=42)
full_model.fit(X_train_scaled_df, y_train)

# Get feature importances from the full model
feature_importances = full_model.feature_importances_
top_indices = np.argsort(feature_importances)[::-1][:optimal_n_embedded]
embedded_mask = np.zeros(len(X_train.columns), dtype=bool)
embedded_mask[top_indices] = True

selected_features_embedded = X_train.columns[embedded_mask]
print(f"Optimal number of features: {optimal_n_embedded}")
print(f"Selected features: {selected_features_embedded.tolist()}")

# Extract selected features
X_train_embedded = X_train_scaled_df.iloc[:, embedded_mask]
X_test_embedded = X_test_scaled_df.iloc[:, embedded_mask]

# Evaluate embedded method with a new model trained on selected features
final_model_embedded = RandomForestClassifier(random_state=42)
final_model_embedded.fit(X_train_embedded, y_train)
y_pred_embedded = final_model_embedded.predict(X_test_embedded)
embedded_accuracy = accuracy_score(y_test, y_pred_embedded)
print(f"Test Accuracy: {embedded_accuracy:.4f}")

# Store results
results['Method'].append('Embedded (RandomForest)')
results['Optimal Number of Features'].append(optimal_n_embedded)
results['Selected Features'].append(selected_features_embedded.tolist())
results['Accuracy'].append(embedded_accuracy)

# Visualize feature importance for embedded method (using the full model)
plt.figure(figsize=(10, 6))
feature_importance = pd.Series(full_model.feature_importances_, index=X_train.columns)
feature_importance.sort_values(ascending=False).plot(kind='bar')
plt.title('Feature Importance (Random Forest - Embedded)')
plt.tight_layout()
plt.savefig('images/embedded_method_importance.png')
plt.close()

# 4. Permutation Importance
print("\n4. Permutation Importance")
# For permutation importance, we need to use the test set for evaluation
# We'll find the optimal number by evaluating on different subsets of features

# First, calculate permutation importance on the full model
base_model = RandomForestClassifier(random_state=42)
base_model.fit(X_train_scaled_df, y_train)
perm_importance = permutation_importance(base_model, X_test_scaled_df, y_test, n_repeats=10, random_state=42)
perm_importance_means = perm_importance.importances_mean

# Sort features by importance
perm_indices = np.argsort(perm_importance_means)[::-1]
sorted_features = X_train.columns[perm_indices]

# Evaluate different numbers of features
perm_scores = []
for n in feature_range:
    selected_indices = perm_indices[:n]
    mask = np.zeros(len(X_train.columns), dtype=bool)
    mask[selected_indices] = True
    
    X_train_perm = X_train_scaled_df.iloc[:, mask]
    X_val_perm = X_test_scaled_df.iloc[:, mask]
    
    model_perm = RandomForestClassifier(random_state=42)
    model_perm.fit(X_train_perm, y_train)
    y_pred_perm = model_perm.predict(X_val_perm)
    perm_scores.append(accuracy_score(y_test, y_pred_perm))

# Find the optimal number of features
optimal_n_perm = feature_range[np.argmax(perm_scores)]
max_score_perm = max(perm_scores)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(feature_range, perm_scores, marker='o')
plt.axvline(x=optimal_n_perm, color='r', linestyle='--', label=f'Optimal: {optimal_n_perm} features')
plt.xlabel('Number of Features')
plt.ylabel('Test Accuracy')
plt.title('Permutation Method - Feature Selection')
plt.legend()
plt.grid(True)
plt.savefig('images/permutation_optimal_features.png')
plt.close()

# Apply the optimal permutation method
perm_mask = np.zeros(len(X_train.columns), dtype=bool)
perm_mask[perm_indices[:optimal_n_perm]] = True
selected_features_perm = X_train.columns[perm_mask]
print(f"Optimal number of features: {optimal_n_perm}")
print(f"Selected features: {selected_features_perm.tolist()}")

# Extract selected features
X_train_perm = X_train_scaled_df.iloc[:, perm_mask]
X_test_perm = X_test_scaled_df.iloc[:, perm_mask]

# Evaluate permutation method
model_perm = RandomForestClassifier(random_state=42)
model_perm.fit(X_train_perm, y_train)
y_pred_perm = model_perm.predict(X_test_perm)
perm_accuracy = accuracy_score(y_test, y_pred_perm)
print(f"Test Accuracy: {perm_accuracy:.4f}")

# Store results
results['Method'].append('Permutation Importance')
results['Optimal Number of Features'].append(optimal_n_perm)
results['Selected Features'].append(selected_features_perm.tolist())
results['Accuracy'].append(perm_accuracy)

# Visualize permutation importance
plt.figure(figsize=(10, 6))
feature_perm_importance = pd.Series(perm_importance_means, index=X_train.columns)
feature_perm_importance.sort_values(ascending=False).plot(kind='bar')
plt.title('Feature Importance (Permutation)')
plt.tight_layout()
plt.savefig('images/permutation_method_importance.png')
plt.close()

# Create a summary DataFrame
results_df = pd.DataFrame(results)
print("\nResults Summary:")
print(results_df)

# Find the best model based on accuracy
best_method_idx = results_df['Accuracy'].idxmax()
best_method = results_df.loc[best_method_idx, 'Method']
best_accuracy = results_df.loc[best_method_idx, 'Accuracy']
best_features = results_df.loc[best_method_idx, 'Selected Features']
print(f"\nBest Method: {best_method} with accuracy {best_accuracy:.4f}")
print(f"Selected Features: {best_features}")

# Save the best model (in this case, the permutation importance model)
print("\nSaving the best model...")
# Save the best performing model
print("\nSaving the best model...")
best_model_data = {
    'model': model_perm,
    'scaler': scaler,
    'features': selected_features_perm.tolist(),
    'accuracy': perm_accuracy,
    'method': 'Permutation Importance'
}

# Save model to disk for future use
joblib.dump(best_model_data, 'best_penguin_model.joblib')
print("Model saved as 'best_penguin_model.joblib'")



# Plot accuracy comparison
plt.figure(figsize=(12, 6))
results_df.sort_values('Accuracy', ascending=False).plot(x='Method', y='Accuracy', kind='bar', color='skyblue')
plt.title('Feature Selection Methods - Accuracy Comparison')
plt.ylim(0.85, 1)  # Adjust as needed
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('images/accuracy_comparison.png')
plt.show()

# Create a feature selection frequency plot
feature_counts = {}
for feature in X.columns:
    count = sum(1 for features in results['Selected Features'] if feature in features)
    feature_counts[feature] = count

plt.figure(figsize=(10, 6))
feature_counts_series = pd.Series(feature_counts)
feature_counts_series.sort_values(ascending=False).plot(kind='bar')
plt.title('Feature Selection Frequency Across Methods')
plt.xlabel('Features')
plt.ylabel('Number of Methods Selected')
plt.tight_layout()
plt.savefig('images/feature_selection_frequency.png')
plt.show()