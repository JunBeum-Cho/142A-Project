import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import roc_curve, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import json
import os

df = pd.read_csv('dataset/conversion.csv')

# Step 1: Data Preprocessing
# Drop irrelevant or confidential features
df = df.drop(['CustomerID', 'AdvertisingPlatform', 'AdvertisingTool'], axis=1)

# Define features
numerical_features = ['Age', 'Income', 'AdSpend', 'ClickThroughRate', 'ConversionRate',
                     'WebsiteVisits', 'PagesPerVisit', 'TimeOnSite', 'SocialShares',
                     'EmailOpens', 'EmailClicks', 'PreviousPurchases', 'LoyaltyPoints']
categorical_features = ['Gender', 'CampaignChannel', 'CampaignType']

# Handle missing values (if any)
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')
df[numerical_features] = num_imputer.fit_transform(df[numerical_features])
df[categorical_features] = cat_imputer.fit_transform(df[categorical_features])

# Encode categorical variables
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# Split features and target
X = df.drop('Conversion', axis=1)
y = df['Conversion']

# Scale numerical features
scaler = StandardScaler()
X[X.columns] = scaler.fit_transform(X)

# Step 2: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Model Training and Evaluation
models = {
    'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced'),
    'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
    'SVM': SVC(random_state=42, probability=True, class_weight='balanced')
}

results = []
for name, model in models.items():
    # Train model
    model.fit(X_train, y_train)
    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)
    # Evaluate (handle cases where metrics may be undefined)
    try:
        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1-Score': f1_score(y_test, y_pred, zero_division=0),
            'ROC-AUC': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0
        })
    except ValueError:
        print(f"Metrics for {name} could not be computed due to single-class test set.")

# Step 3.5: Initial Feature Selection for Random Forest Optimization (by Importance Threshold)
print("\n--- Initial Feature Selection for RF Optimization (Importance >= 0.05) ---")
# Train a default RF model on all features to get initial importances
initial_rf = RandomForestClassifier(random_state=42, class_weight='balanced')
initial_rf.fit(X_train, y_train)
initial_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': initial_rf.feature_importances_
}).sort_values('Importance', ascending=False)

# Filter features with importance >= 0.05
high_importance_features = initial_importances[initial_importances['Importance'] >= 0.05]
high_importance_feature_names = high_importance_features['Feature'].tolist()

if not high_importance_feature_names:
    print("No features found with importance >= 0.05. Using all features for optimization instead.")
    # Fallback to using all features if none meet the threshold, or you could choose top N as a fallback.
    # For this example, we proceed with an empty list, which will likely cause issues downstream.
    # A better fallback would be to use X_train.columns or top_N_features.
    # For now, let's make it select all features if none meet criteria to avoid empty X_train_selected.
    if initial_importances.empty:
        print("Warning: Initial feature importances are empty. Using all original features.")
        high_importance_feature_names = X_train.columns.tolist()
    else:
        print("No features met importance >= 0.05. Falling back to top 5 features to avoid empty set.")
        high_importance_feature_names = initial_importances.head(5)['Feature'].tolist()
        if not high_importance_feature_names:
             print("Fallback to top 5 also resulted in no features. Using all original features.")
             high_importance_feature_names = X_train.columns.tolist()

print(f"Features selected with importance >= 0.05 (or fallback): {high_importance_feature_names}")

# Create training and testing sets with only these selected features
X_train_selected = X_train[high_importance_feature_names]
X_test_selected = X_test[high_importance_feature_names]

# Step 4: Optimize Best Model (Random Forest) using Selected Features
print(f"\n--- Optimizing Random Forest using {len(high_importance_feature_names)} selected features ---")
# param_grid_rf = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [3, 5, 10, None],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
# }
param_grid_rf = {
  'max_depth': [10],
  'min_samples_leaf': [4],
  'min_samples_split': [10],
  'n_estimators': [300]
}
rf_classifier_selected = RandomForestClassifier(random_state=42, class_weight='balanced')
grid_search_rf = GridSearchCV(rf_classifier_selected, param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)

if X_train_selected.empty:
    print("Skipping GridSearchCV as no features were selected.")
    best_model = rf_classifier_selected # or handle error appropriately
else:
    grid_search_rf.fit(X_train_selected, y_train) # Fit on data with selected features
    best_model = grid_search_rf.best_estimator_
    print(f"\nBest Random Forest Parameters (trained on selected features): {grid_search_rf.best_params_}")

# Step 5: Final Evaluation (with Optimized Random Forest on Selected Features)
# Evaluate on the X_test_selected
if X_test_selected.empty and not X_train_selected.empty : # if train was not empty but test is (e.g. all features dropped)
    print("Cannot evaluate model as X_test_selected is empty due to feature selection.")
    accuracy_selected = 0
    precision_selected = 0
    recall_selected = 0
    f1_selected = 0
    roc_auc_selected = 0
elif X_train_selected.empty: # If training itself was skipped
    print("Cannot evaluate model as training was skipped due to no features.")
    accuracy_selected, precision_selected, recall_selected, f1_selected, roc_auc_selected = (0,0,0,0,0)
else:
    y_pred = best_model.predict(X_test_selected)
    y_proba = best_model.predict_proba(X_test_selected)[:, 1]
    accuracy_selected = accuracy_score(y_test, y_pred)
    precision_selected = precision_score(y_test, y_pred, zero_division=0)
    recall_selected = recall_score(y_test, y_pred, zero_division=0)
    f1_selected = f1_score(y_test, y_pred, zero_division=0)
    roc_auc_selected = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0

try:
    final_results = {
        'Model': 'Optimized RF (Feat Imp >= 0.05)',
        'Accuracy': accuracy_selected,
        'Precision': precision_selected,
        'Recall': recall_selected,
        'F1-Score': f1_selected,
        'ROC-AUC': roc_auc_selected
    }
    results.append(final_results)
except ValueError:
    print("Final metrics for Optimized Random Forest (Selected Features) could not be computed.")

# Step 6: Feature Importance (from Optimized Random Forest on Selected Features)
if not X_train_selected.empty:
    feature_importance = pd.DataFrame({
        'Feature': X_train_selected.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
else:
    print("Skipping feature importance calculation as no features were selected for training.")
    feature_importance = pd.DataFrame(columns=['Feature', 'Importance']) # empty dataframe

# Step 7: Output Results
print("\nModel Comparison:")
results_df = pd.DataFrame(results)
print(results_df)
print("\nFeature Importance (Optimized Random Forest):")
print(feature_importance)

# Step 8: Visualizations
print("\n--- Generating Visualizations ---")

# 1. Model Performance Comparison
if not results_df.empty:
    # Ensure 'Accuracy' column exists and sort by it for better visualization
    if 'Accuracy' in results_df.columns:
        results_df_sorted = results_df.sort_values(by='Accuracy', ascending=False)
        plt.figure(figsize=(10, 6))
        plt.bar(results_df_sorted['Model'], results_df_sorted['Accuracy'], color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'])
        plt.xlabel("Model")
        plt.ylabel("Accuracy")
        plt.title("Model Accuracy Comparison")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
    else:
        print("Skipping model performance plot: 'Accuracy' column not found in results_df.")
else:
    print("Skipping model performance plot: results_df is empty.")

# 2. Feature Importance Plot
if not feature_importance.empty:
    # Take top N features for cleaner plot, e.g., top 10 or all if less than 10
    top_n = min(len(feature_importance), 10)
    feature_importance_top_n = feature_importance.head(top_n)
    
    plt.figure(figsize=(10, max(6, top_n * 0.5))) # Adjust height based on number of features
    plt.barh(feature_importance_top_n['Feature'], feature_importance_top_n['Importance'], color='teal')
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Feature Importances (Optimized Random Forest)")
    plt.gca().invert_yaxis() # Display most important at the top
    plt.tight_layout()
    plt.show()
else:
    print("Skipping feature importance plot: feature_importance DataFrame is empty.")

# 3. ROC Curve for the optimized model (Optional, and if applicable)
# Check if necessary data for ROC curve is available and meaningful
if 'roc_auc_selected' in locals() and roc_auc_selected > 0 and 'y_proba' in locals() and y_test is not None:
    fpr, tpr, thresholds = roc_curve(y_test, y_proba) # y_proba from Step 5
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_selected:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Optimized RF)')
    plt.legend(loc="lower right")
    plt.show()
else:
    print("Skipping ROC curve plot: Conditions not met (e.g., ROC-AUC is 0, y_proba not available, or y_test is None).")

# 4. Confusion Matrix for the optimized model
if 'y_pred' in locals() and y_test is not None and not X_train_selected.empty:
    cm = confusion_matrix(y_test, y_pred) # y_pred from Step 5
    plt.figure(figsize=(6, 5))
    # Using matshow to display the matrix, and text to annotate cells
    plt.matshow(cm, cmap=plt.cm.Blues, alpha=0.3, fignum=0) # fignum=0 reuses the figure created by plt.figure
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='large')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix (Optimized RF)')
    plt.xticks([0, 1], ['Predicted 0', 'Predicted 1'])
    plt.yticks([0, 1], ['True 0', 'True 1'])
    plt.show()
else:
    print("Skipping Confusion Matrix plot: Conditions not met (e.g., y_pred not available or training was skipped).")

# Step 9: Save results to JSON
print("\n--- Saving results to dataset/result.json ---")
output_data = {}

# Model Comparison
if 'results_df' in locals() and isinstance(results_df, pd.DataFrame):
    output_data["model_comparison"] = results_df.to_dict(orient="records")
else:
    output_data["model_comparison"] = []

# Feature Importance
if 'feature_importance' in locals() and isinstance(feature_importance, pd.DataFrame):
    output_data["feature_importance"] = feature_importance.to_dict(orient="records")
else:
    output_data["feature_importance"] = []

# Data for Optimized RF charts
if not X_train_selected.empty: # Check if optimized model was trained
    # ROC Curve data
    if 'roc_auc_selected' in locals() and 'fpr' in locals() and 'tpr' in locals() and isinstance(fpr, np.ndarray) and isinstance(tpr, np.ndarray):
        output_data["roc_curve_data"] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": roc_auc_selected if 'roc_auc_selected' in locals() else None
        }
    
    # Confusion Matrix data
    if 'cm' in locals() and isinstance(cm, np.ndarray): # cm is calculated in existing visualization step 4
        output_data["confusion_matrix_data"] = {
            "matrix": cm.tolist(),
            "labels": ["Actual Negative", "Actual Positive"], # Assuming class 0 and 1
            "predicted_labels": ["Predicted Negative", "Predicted Positive"]
        }

    # Precision-Recall Curve data
    if 'precision' in locals() and 'recall' in locals() and isinstance(precision, np.ndarray) and isinstance(recall, np.ndarray): # precision, recall from existing visualization step 5
        output_data["precision_recall_data"] = {
            "precision": precision.tolist(),
            "recall": recall.tolist()
        }

    # Predicted Probabilities data (raw and for histogram)
    if 'y_proba' in locals() and isinstance(y_proba, np.ndarray): # y_proba from Step 5
        output_data["predicted_probabilities_data"] = {
            "raw_probabilities": y_proba.tolist()
        }
        # Calculate histogram data
        hist_counts, bin_edges = np.histogram(y_proba, bins=10) # 10 bins
        output_data["probability_histogram_data"] = {
            "counts": hist_counts.tolist(),
            "bin_edges": bin_edges.tolist() 
            # For plotting, bin labels can be like f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}"
        }
else:
    print("Optimized model training was skipped, so detailed chart data for it will not be saved.")


# Ensure output directory exists
# Save into src/dashborad/src/app/ for direct import by page.tsx
# This assumes src/index.py is in project_root/src/ and dashboard is project_root/src/dashborad/
output_dir = os.path.join(os.path.dirname(__file__), "..", "src", "dashborad", "src", "app")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_path = os.path.join(output_dir, "result.json")
try:
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"Results successfully saved to {output_path}")
except Exception as e:
    print(f"Error saving results to JSON: {e}")
