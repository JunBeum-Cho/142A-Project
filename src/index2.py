import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Separate features and target
df = pd.read_csv('src/dataset.csv')

X = df.drop(['CustomerID', 'Conversion'], axis=1)
y = df['Conversion']

# Define categorical and numerical columns
cat_cols = ['Gender', 'CampaignChannel', 'CampaignType', 'AdvertisingPlatform', 'AdvertisingTool']
num_cols = ['Age', 'Income', 'AdSpend', 'ClickThroughRate', 'ConversionRate', 'WebsiteVisits',
            'PagesPerVisit', 'TimeOnSite', 'SocialShares', 'EmailOpens', 'EmailClicks',
            'PreviousPurchases', 'LoyaltyPoints']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='mean')),
                          ('scaler', StandardScaler())]), num_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                          ('encoder', OneHotEncoder(drop='first', sparse_output=False))]), cat_cols)
    ])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Full pipeline with Gradient Boosting
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

# Cross-validation
scores = cross_val_score(pipeline, X_train, y_train, scoring='roc_auc', cv=5)
print(f"Mean ROC-AUC: {scores.mean():.3f}")

# Hyperparameter tuning
param_grid = {'classifier__n_estimators': [100],
              'classifier__learning_rate': [0.1],
              'classifier__max_depth': [3]}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)

# Final evaluation
best_model = grid_search.best_estimator_
test_score = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
print(f"Test ROC-AUC: {test_score:.3f}")

# Covariance Matrix
X_train_processed = pd.DataFrame(best_model.named_steps['preprocessor'].transform(X_train),
                                 columns=best_model.named_steps['preprocessor'].get_feature_names_out())

# Get the names of the processed numerical columns
processed_num_cols = [col for col in X_train_processed.columns if col.startswith('num__')]

cov_matrix = X_train_processed[processed_num_cols].cov()
print("\nCovariance Matrix of Numerical Features:")
print(cov_matrix)

# VIF for numerical features
# Ensure we are using the scaled numerical features from the training set
numerical_features_processed = X_train_processed[processed_num_cols]

vif_data = pd.DataFrame()
# Adjust feature names for VIF output to be more readable (remove 'num__')
vif_data["feature"] = [col.replace('num__', '') for col in numerical_features_processed.columns]
vif_data["VIF"] = [variance_inflation_factor(numerical_features_processed.values, i)
                     for i in range(len(numerical_features_processed.columns))]
print("\nVIF for Numerical Features:")
print(vif_data)

# Feature Importance
feature_importances = best_model.named_steps['classifier'].feature_importances_
feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
importances_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importances})
importances_df = importances_df.sort_values(by='importance', ascending=False)
print("\nFeature Importances:")
print(importances_df)

# --- Retrain with selected features ---
print("\n--- Retraining with selected features ---")

# 1. Identify features with VIF > 5
high_vif_features = vif_data[vif_data['VIF'] > 5]['feature'].tolist()
print(f"Numerical features with VIF > 5: {high_vif_features}")

# 2. Identify top 10 most important features (original names)
top_10_features_processed = importances_df.head(12)['feature'].tolist()
top_10_features_original = []
for feature_name in top_10_features_processed:
    if feature_name.startswith('num__'):
        top_10_features_original.append(feature_name.replace('num__', ''))
    elif feature_name.startswith('cat__'):
        # For one-hot encoded features, take the original categorical column name
        # e.g., cat__Gender_Male -> Gender
        original_cat_name = feature_name.split('__')[1].split('_')[0]
        if original_cat_name not in top_10_features_original: # Avoid duplicates if multiple encoded values from same original col are in top 10
            top_10_features_original.append(original_cat_name)
    else: # Should not happen with ColumnTransformer default naming
        top_10_features_original.append(feature_name)


print(f"Top 10 most important features (original names): {top_10_features_original}")

# 3. Combine feature lists
selected_features_original = list(set(high_vif_features + top_10_features_original))
print(f"Combined selected features for retraining: {selected_features_original}")

# Separate selected features into new numerical and categorical lists
new_num_cols = [col for col in selected_features_original if col in num_cols]
new_cat_cols = [col for col in selected_features_original if col in cat_cols]

print(f"New numerical columns for retraining: {new_num_cols}")
print(f"New categorical columns for retraining: {new_cat_cols}")


if not new_num_cols and not new_cat_cols:
    print("\nNo features selected based on VIF > 5 or top 10 importance. Skipping retraining.")
else:
    # 4. Create new preprocessor and pipeline
    # Ensure transformers are only added if there are columns for them
    new_transformers = []
    if new_num_cols:
        new_transformers.append(
            ('num', Pipeline([('imputer', SimpleImputer(strategy='mean')),
                              ('scaler', StandardScaler())]), new_num_cols)
        )
    if new_cat_cols:
        new_transformers.append(
            ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                              ('encoder', OneHotEncoder(drop='first', sparse_output=False))]), new_cat_cols)
        )

    if not new_transformers:
        print("\nNo numerical or categorical features selected for the new preprocessor. Skipping retraining.")
    else:
        new_preprocessor = ColumnTransformer(transformers=new_transformers)

        new_pipeline = Pipeline([
            ('preprocessor', new_preprocessor),
            ('classifier', GradientBoostingClassifier(random_state=42))
        ])

        # 5. Retrain GridSearchCV
        # Using the same param_grid as before
        new_grid_search = GridSearchCV(new_pipeline, param_grid, cv=5, scoring='roc_auc')
        new_grid_search.fit(X_train, y_train) # Fit on original X_train, preprocessor will select columns

        # 6. Evaluate the new model
        best_model_new = new_grid_search.best_estimator_

        # Check if preprocessor produced any output features before predicting
        try:
            # Attempt to transform to see if there are output features
            X_test_transformed_check = best_model_new.named_steps['preprocessor'].transform(X_test)
            if X_test_transformed_check.shape[1] == 0:
                 print("\nPreprocessor in the new model produced no output features. Cannot evaluate.")
                 test_score_new = np.nan
            else:
                test_score_new = roc_auc_score(y_test, best_model_new.predict_proba(X_test)[:, 1])
                print(f"\nNew Test ROC-AUC (with selected features): {test_score_new:.3f}")
        except ValueError as e:
            print(f"\nError during prediction with the new model (possibly due to no features): {e}")
            test_score_new = np.nan


        # 7. Print comparison
        print(f"Original Test ROC-AUC: {test_score:.3f}")
        if not np.isnan(test_score_new):
            print(f"Difference (New - Original): {test_score_new - test_score:.3f}")