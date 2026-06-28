import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure the plots directory exists
os.makedirs('plots', exist_ok=True)

# ----------------------------------------------------
# 1. DATA LOADING
# ----------------------------------------------------
print("--- Step 1: Loading Dataset ---")
df = pd.read_csv('car_data.csv')
print(f"Dataset shape: {df.shape}")
print("First 5 rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())

# ----------------------------------------------------
# 2. FEATURE ENGINEERING
# ----------------------------------------------------
print("\n--- Step 2: Feature Engineering ---")

# Feature 1: Car Age (Current Year 2026 - Manufacturing Year)
CURRENT_YEAR = 2026
df['Car_Age'] = CURRENT_YEAR - df['Year']
print(f"Calculated 'Car_Age' using current year {CURRENT_YEAR}.")

# Feature 2: Brand Mapping & Brand Goodwill (Tiering)
def get_brand(car_name):
    name = car_name.lower().strip()
    if any(x in name for x in ['fortuner', 'innova', 'corolla', 'etios', 'camry', 'land cruiser']):
        return 'Toyota'
    elif any(x in name for x in ['brio', 'amaze', 'city', 'jazz', 'accord', 'civic']):
        return 'Honda'
    elif any(x in name for x in ['i10', 'i20', 'eon', 'xcent', 'elantra', 'creta', 'verna', 'grand i10']):
        return 'Hyundai'
    elif any(x in name for x in ['ritz', 'sx4', 'ciaz', 'wagon r', 'swift', 'vitara brezza', 's cross', 'alto', 'ertiga', 'dzire', 'ignis', 'baleno', 'omni', '800']):
        return 'Maruti Suzuki'
    else:
        return 'Other'

df['Brand'] = df['Car_Name'].apply(get_brand)

def get_goodwill_tier(brand):
    if brand in ['Toyota', 'Honda']:
        return 'High_Goodwill'
    elif brand == 'Hyundai':
        return 'Medium_Goodwill'
    else:
        return 'Budget_Goodwill'

df['Brand_Goodwill'] = df['Brand'].apply(get_goodwill_tier)

print("Brand mapping counts:")
print(df['Brand'].value_counts())
print("\nBrand Goodwill Tier counts:")
print(df['Brand_Goodwill'].value_counts())

# ----------------------------------------------------
# 3. DATA PREPROCESSING
# ----------------------------------------------------
print("\n--- Step 3: Data Preprocessing ---")

# Let's check for missing values
print("Missing values in each column:")
print(df.isnull().sum())

# Drop the original Car_Name and Brand columns as they are high-cardinality/redundant now
# and the Year column because we have Car_Age.
model_df = df.drop(columns=['Car_Name', 'Brand', 'Year'])

# One-hot encode categorical features: Fuel_Type, Seller_Type, Transmission, Brand_Goodwill
# Using drop_first=True to avoid multi-collinearity (the dummy variable trap)
model_df = pd.get_dummies(model_df, columns=['Fuel_Type', 'Seller_Type', 'Transmission', 'Brand_Goodwill'], drop_first=True)

# Convert boolean dummy columns to 0/1 integers for Scikit-Learn compatibility
bool_cols = model_df.select_dtypes(include=['bool']).columns
model_df[bool_cols] = model_df[bool_cols].astype(int)

print(f"Preprocessed dataset columns: {list(model_df.columns)}")
print("Preprocessed dataset head:")
print(model_df.head())

# Split into Features (X) and Target (y)
X = model_df.drop(columns=['Selling_Price'])
y = model_df['Selling_Price']

# Split into Training and Testing Sets (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# ----------------------------------------------------
# 4. MODEL TRAINING
# ----------------------------------------------------
print("\n--- Step 4: Model Training ---")

# Model A: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Model B: Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Models trained successfully!")

# ----------------------------------------------------
# 5. MODEL EVALUATION
# ----------------------------------------------------
print("\n--- Step 5: Model Evaluation ---")

def evaluate_predictions(y_true, y_pred, model_name):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"[{model_name}] R-squared (R2) Score: {r2:.4f}")
    print(f"[{model_name}] Mean Absolute Error (MAE): {mae:.4f} Lakhs")
    print(f"[{model_name}] Root Mean Squared Error (RMSE): {rmse:.4f} Lakhs")
    return {"R2": r2, "MAE": mae, "RMSE": rmse}

print("--- Linear Regression Performance ---")
lr_metrics = evaluate_predictions(y_test, y_pred_lr, "Linear Regression")

print("\n--- Random Forest Regressor Performance ---")
rf_metrics = evaluate_predictions(y_test, y_pred_rf, "Random Forest")

# Check which model is better
best_model_name = "Random Forest" if rf_metrics["R2"] > lr_metrics["R2"] else "Linear Regression"
best_y_pred = y_pred_rf if best_model_name == "Random Forest" else y_pred_lr
print(f"\nBest Model based on R2 Score: {best_model_name}")

# ----------------------------------------------------
# 6. DATA VISUALIZATION
# ----------------------------------------------------
print("\n--- Step 6: Generating & Saving Plots ---")

# Set aesthetic styling
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# Plot 1: Correlation Heatmap (of original numerical features + engineered features)
plt.figure(figsize=(8, 6))
numeric_cols = df.select_dtypes(include=[np.number]).copy()
numeric_cols['Car_Age'] = df['Car_Age']
# Exclude Year from correlation heatmap as Car_Age is its direct replacement
if 'Year' in numeric_cols.columns:
    numeric_cols = numeric_cols.drop(columns=['Year'])
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Numeric Features", fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('plots/correlation_heatmap.png', dpi=300)
plt.close()
print("Saved correlation heatmap.")

# Plot 2: Brand Goodwill Tier vs Price Distribution
plt.figure(figsize=(8, 6))
sns.boxplot(x='Brand_Goodwill', y='Selling_Price', data=df, hue='Brand_Goodwill', palette='Set2', legend=False)
plt.title("Car Selling Price by Brand Goodwill Tier", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Brand Goodwill Tier", fontsize=12)
plt.ylabel("Selling Price (in Lakhs)", fontsize=12)
plt.tight_layout()
plt.savefig('plots/brand_price_distribution.png', dpi=300)
plt.close()
print("Saved brand price distribution boxplot.")

# Plot 3: Actual vs Predicted Prices (for the best model)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, best_y_pred, alpha=0.7, color='#2c3e50', edgecolors='w', s=70)
# Draw perfect prediction line
min_val = min(min(y_test), min(best_y_pred))
max_val = max(max(y_test), max(best_y_pred))
plt.plot([min_val, max_val], [min_val, max_val], color='#e74c3c', linestyle='--', linewidth=2, label='Perfect Prediction')
plt.title(f"Actual vs. Predicted Prices ({best_model_name})", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Actual Selling Price (in Lakhs)", fontsize=12)
plt.ylabel("Predicted Selling Price (in Lakhs)", fontsize=12)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('plots/actual_vs_predicted.png', dpi=300)
plt.close()
print("Saved actual vs predicted scatter plot.")

# Plot 4: Feature Importance (from Random Forest)
plt.figure(figsize=(10, 6))
importances = rf_model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

sns.barplot(x=importances[indices], y=feature_names[indices], palette='viridis', hue=feature_names[indices], legend=False)
plt.title("Feature Importance Analysis (Random Forest)", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Relative Importance Score", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.tight_layout()
plt.savefig('plots/feature_importance.png', dpi=300)
plt.close()
print("Saved feature importance plot.")

print("\n--- Pipeline Completed Successfully! ---")
