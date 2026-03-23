import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')   # Hide unnecessary warnings

print("=" * 55)
print("   CARDIOVASCULAR DISEASE PREDICTION PROJECT")
print("=" * 55)

# STEP 2: LOAD THE DATASET
# NOTE: Change the filename below to match YOUR file name.
# Common separators: ',' for .csv, ';' for some datasets.

print("\n[STEP 2] Loading dataset...")

# Try comma separator first, then semicolon
try:
    df = pd.read_csv("cardio_train.csv", sep=';')
    if df.shape[1] == 1:          # If only 1 column, wrong separator
        df = pd.read_csv("cardio_train.csv", sep=',')
except FileNotFoundError:
    # If file name is different, try this:
    df = pd.read_csv("cardio_train.csv")   #CHANGE filename if needed

print(f"   Dataset loaded! Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\n   First 5 rows:")
print(df.head())

# STEP 3: DATA PRE-PROCESSING

print("\n[STEP 3] Data Pre-Processing...")

#3a. Drop 'id' column (not useful for prediction) ---
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)
    print("   ✓ Dropped 'id' column")

#3b. Convert age from days → years (if in days) ---
# Age values like 18000-25000 mean it's in DAYS
if df['age'].max() > 200:
    df['age'] = (df['age'] / 365).round(1)
    print("   ✓ Converted age from days to years")

#3c. Check for missing values ---
print(f"\n   Missing values per column:")
print(df.isnull().sum())

# Fill missing values if any (most medical datasets are clean)
df.dropna(inplace=True)
print("   ✓ Removed rows with missing values (if any)")

# --- 3d. Remove outliers in blood pressure ---
# Blood pressure cannot be negative or extremely high
df = df[df['ap_hi'] > 0]
df = df[df['ap_lo'] > 0]
df = df[df['ap_hi'] < 250]
df = df[df['ap_lo'] < 200]
df = df[df['ap_hi'] >= df['ap_lo']]   # Systolic must be >= Diastolic
print("   ✓ Removed blood pressure outliers")

# --- 3e. Remove outliers in height and weight ---
df = df[(df['height'] > 100) & (df['height'] < 250)]
df = df[(df['weight'] > 30) & (df['weight'] < 200)]
print("   ✓ Removed height/weight outliers")

# --- 3f. Add BMI feature ---
df['bmi'] = (df['weight'] / ((df['height'] / 100) ** 2)).round(2)
print("   ✓ Added BMI (Body Mass Index) feature")

print(f"\n   Clean dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\n   Basic Statistics:")
print(df.describe().round(2))

# STEP 4: DATA ANALYSIS & VISUALIZATIONS

print("\n[STEP 4] Creating Visualizations...")

# Set a nice style for all plots
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)


#PLOT 1: Cardio Disease Count (Target Distribution) ---
plt.figure(figsize=(7, 5))
counts = df['cardio'].value_counts()
bars = plt.bar(['No Disease (0)', 'Has Disease (1)'], counts.values,
               color=['#2ecc71', '#e74c3c'], edgecolor='black', width=0.5)
for bar, count in zip(bars, counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
             f'{count}\n({count/len(df)*100:.1f}%)',
             ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.title('Distribution of Cardiovascular Disease (Target Variable)', fontsize=14, fontweight='bold')
plt.ylabel('Number of Patients')
plt.xlabel('Cardio Disease')
plt.tight_layout()
plt.savefig('plot1_target_distribution.png', dpi=150)
plt.show()
print("   ✓ Plot 1: Target Distribution saved")


#PLOT 2: Age Distribution by Disease ---
plt.figure(figsize=(10, 5))
df[df['cardio'] == 0]['age'].plot(kind='hist', bins=30, alpha=0.6,
                                   color='#2ecc71', label='No Disease', edgecolor='black')
df[df['cardio'] == 1]['age'].plot(kind='hist', bins=30, alpha=0.6,
                                   color='#e74c3c', label='Has Disease', edgecolor='black')
plt.title('Age Distribution by Cardiovascular Disease', fontsize=14, fontweight='bold')
plt.xlabel('Age (years)')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.savefig('plot2_age_distribution.png', dpi=150)
plt.show()
print("   ✓ Plot 2: Age Distribution saved")


#PLOT 3: Gender vs Cardio Disease (Bar Chart) ---
plt.figure(figsize=(8, 5))
gender_cardio = df.groupby(['gender', 'cardio']).size().unstack()
gender_cardio.index = ['Female (1)', 'Male (2)']
gender_cardio.columns = ['No Disease', 'Has Disease']
gender_cardio.plot(kind='bar', color=['#2ecc71', '#e74c3c'],
                   edgecolor='black', ax=plt.gca())
plt.title('Gender vs Cardiovascular Disease', fontsize=14, fontweight='bold')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend()
plt.tight_layout()
plt.savefig('plot3_gender_vs_cardio.png', dpi=150)
plt.show()
print("   ✓ Plot 3: Gender vs Cardio saved")


#PLOT 4: Cholesterol vs Cardio ---
plt.figure(figsize=(8, 5))
chol_cardio = df.groupby(['cholesterol', 'cardio']).size().unstack()
chol_cardio.index = ['Normal (1)', 'Above Normal (2)', 'Well Above Normal (3)']
chol_cardio.columns = ['No Disease', 'Has Disease']
chol_cardio.plot(kind='bar', color=['#2ecc71', '#e74c3c'],
                 edgecolor='black', ax=plt.gca())
plt.title('Cholesterol Level vs Cardiovascular Disease', fontsize=14, fontweight='bold')
plt.xlabel('Cholesterol Level')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend()
plt.tight_layout()
plt.savefig('plot4_cholesterol_vs_cardio.png', dpi=150)
plt.show()
print("   ✓ Plot 4: Cholesterol vs Cardio saved")


#PLOT 5: Blood Pressure (Systolic) Boxplot ---
plt.figure(figsize=(8, 5))
df.boxplot(column='ap_hi', by='cardio', grid=False,
           boxprops=dict(color='navy'),
           medianprops=dict(color='red', linewidth=2))
plt.suptitle('')  # Remove auto title
plt.title('Systolic Blood Pressure by Cardiovascular Disease', fontsize=14, fontweight='bold')
plt.xlabel('Cardio (0=No Disease, 1=Has Disease)')
plt.ylabel('Systolic BP (ap_hi)')
plt.tight_layout()
plt.savefig('plot5_bp_boxplot.png', dpi=150)
plt.show()
print("   ✓ Plot 5: Blood Pressure Boxplot saved")


#PLOT 6: BMI Distribution by Disease ---
plt.figure(figsize=(10, 5))
df[df['cardio'] == 0]['bmi'].plot(kind='hist', bins=40, alpha=0.6,
                                   color='#3498db', label='No Disease', edgecolor='black')
df[df['cardio'] == 1]['bmi'].plot(kind='hist', bins=40, alpha=0.6,
                                   color='#e74c3c', label='Has Disease', edgecolor='black')
plt.xlim(10, 60)
plt.title('BMI Distribution by Cardiovascular Disease', fontsize=14, fontweight='bold')
plt.xlabel('BMI')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.savefig('plot6_bmi_distribution.png', dpi=150)
plt.show()
print("   ✓ Plot 6: BMI Distribution saved")


#PLOT 7: Lifestyle Factors (Smoke, Alcohol, Active) ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
lifestyle = ['smoke', 'alco', 'active']
titles = ['Smoking vs Cardio', 'Alcohol vs Cardio', 'Physical Activity vs Cardio']

for ax, col, title in zip(axes, lifestyle, titles):
    data = df.groupby([col, 'cardio']).size().unstack()
    data.columns = ['No Disease', 'Has Disease']
    data.plot(kind='bar', color=['#2ecc71', '#e74c3c'],
              edgecolor='black', ax=ax)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(col.capitalize())
    ax.set_ylabel('Count')
    ax.set_xticklabels(['No', 'Yes'], rotation=0)

plt.suptitle('Lifestyle Factors vs Cardiovascular Disease', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot7_lifestyle_factors.png', dpi=150)
plt.show()
print("   ✓ Plot 7: Lifestyle Factors saved")


#PLOT 8: Pairplot (Key Features) ---
print("   Creating pairplot (takes a moment)...")
pair_cols = ['age', 'bmi', 'ap_hi', 'ap_lo', 'cardio']
sample_df = df[pair_cols].sample(n=min(1000, len(df)), random_state=42)
pair_plot = sns.pairplot(sample_df, hue='cardio',
                          palette={0: '#2ecc71', 1: '#e74c3c'},
                          plot_kws={'alpha': 0.5})
pair_plot.fig.suptitle('Pairplot of Key Features', y=1.02, fontsize=14, fontweight='bold')
plt.savefig('plot8_pairplot.png', dpi=100)
plt.show()
print("   ✓ Plot 8: Pairplot saved")

# STEP 5: CORRELATION MATRIX

print("\n[STEP 5] Correlation Matrix...")

plt.figure(figsize=(12, 9))
corr_matrix = df.corr()

# Create heatmap
sns.heatmap(corr_matrix,
            annot=True,           # Show numbers
            fmt='.2f',            # 2 decimal places
            cmap='coolwarm',      # Color: blue=negative, red=positive
            vmin=-1, vmax=1,
            square=True,
            linewidths=0.5,
            linecolor='white',
            annot_kws={'size': 9})

plt.title('Correlation Matrix of All Features', fontsize=15, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig('plot9_correlation_matrix.png', dpi=150)
plt.show()
print("   ✓ Correlation Matrix saved")

# Print top correlations with 'cardio'
print("\n   Top features correlated with 'cardio' (target):")
cardio_corr = corr_matrix['cardio'].drop('cardio').abs().sort_values(ascending=False)
for feature, value in cardio_corr.items():
    print(f"   {feature:15s}: {value:.3f}")

# STEP 6: PREPARE DATA FOR ML MODELS

print("\n[STEP 6] Preparing data for Machine Learning...")

# Separate features (X) and target (y)
X = df.drop('cardio', axis=1)    # All columns except 'cardio'
y = df['cardio']                  # Only 'cardio' column

print(f"   Features (X) shape: {X.shape}")
print(f"   Target  (y) shape: {y.shape}")
print(f"   Feature names: {list(X.columns)}")

# Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,       # 20% for testing
    random_state=42,      # For reproducibility
    stratify=y            # Keep class balance
)
print(f"\n   Training samples : {len(X_train)}")
print(f"   Testing  samples : {len(X_test)}")

# Scale the features (normalize to same range)
# This is important for SVM and KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("   ✓ Feature scaling done (StandardScaler)")

# STEP 7: TRAIN & EVALUATE ALL ML MODELS

print("\n[STEP 7] Training Machine Learning Models...")
print("-" * 55)

# Store results for comparison
results = {}

# MODEL 1: LOGISTIC REGRESSION (LR)
# Simple, fast, good baseline model

print("\n   [1/5] Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_acc  = accuracy_score(y_test, lr_pred)
results['Logistic Regression'] = lr_acc
print(f"        Accuracy: {lr_acc * 100:.2f}%")

# MODEL 2: K-NEAREST NEIGHBOR (KNN)
# Classifies based on k closest data points

print("\n   [2/5] K-Nearest Neighbor (KNN)...")
knn_model = KNeighborsClassifier(n_neighbors=5)   # Try 5 neighbors
knn_model.fit(X_train_scaled, y_train)
knn_pred = knn_model.predict(X_test_scaled)
knn_acc  = accuracy_score(y_test, knn_pred)
results['KNN'] = knn_acc
print(f"        Accuracy: {knn_acc * 100:.2f}%")

# MODEL 3: SUPPORT VECTOR MACHINE (SVM)
# Finds the best boundary between classes

print("\n   [3/5] Support Vector Machine (SVM)...")
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)
svm_acc  = accuracy_score(y_test, svm_pred)
results['SVM'] = svm_acc
print(f"        Accuracy: {svm_acc * 100:.2f}%")

# MODEL 4: DECISION TREE (DT)
# Tree-like flowchart of decisions

print("\n   [4/5] Decision Tree...")
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)   # No scaling needed for trees
dt_pred = dt_model.predict(X_test)
dt_acc  = accuracy_score(y_test, dt_pred)
results['Decision Tree'] = dt_acc
print(f"        Accuracy: {dt_acc * 100:.2f}%")

# MODEL 5: RANDOM FOREST (RF)
# Many trees combined = stronger model

print("\n   [5/5] Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
rf_model.fit(X_train, y_train)   # No scaling needed for trees
rf_pred = rf_model.predict(X_test)
rf_acc  = accuracy_score(y_test, rf_pred)
results['Random Forest'] = rf_acc
print(f"        Accuracy: {rf_acc * 100:.2f}%")

# STEP 8: COMPARE ALL MODELS

print("\n[STEP 8] Model Comparison...")
print("\n" + "=" * 45)
print(f"   {'Model':<22} | {'Accuracy':>8}")
print("=" * 45)
for model_name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    bar = '█' * int(acc * 30)
    print(f"   {model_name:<22} | {acc*100:>7.2f}%")
print("=" * 45)

best_model_name = max(results, key=results.get)
best_acc = results[best_model_name]
print(f"\n   🏆 BEST MODEL: {best_model_name} with {best_acc*100:.2f}% accuracy")


#PLOT: Model Accuracy Comparison Bar Chart
plt.figure(figsize=(10, 6))
model_names = list(results.keys())
accuracies  = [v * 100 for v in results.values()]
colors = ['#e74c3c' if name == best_model_name else '#3498db' for name in model_names]

bars = plt.bar(model_names, accuracies, color=colors, edgecolor='black', width=0.5)
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.ylim(min(accuracies) - 5, 100)
plt.title('Accuracy Comparison of ML Models', fontsize=14, fontweight='bold')
plt.xlabel('Machine Learning Model')
plt.ylabel('Accuracy (%)')
plt.xticks(rotation=10)
plt.axhline(y=max(accuracies), color='red', linestyle='--', alpha=0.5, label='Best Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('plot10_model_comparison.png', dpi=150)
plt.show()
print("   ✓ Model Comparison chart saved")

# STEP 9: DETAILED REPORT FOR BEST MODEL

print(f"\n[STEP 9] Detailed Report for Best Model: {best_model_name}")
print("-" * 55)

# Pick predictions for best model
best_preds_map = {
    'Logistic Regression': lr_pred,
    'KNN':                 knn_pred,
    'SVM':                 svm_pred,
    'Decision Tree':       dt_pred,
    'Random Forest':       rf_pred
}
best_pred = best_preds_map[best_model_name]

# Classification Report
print(f"\n   Classification Report:")
print(classification_report(y_test, best_pred,
                             target_names=['No Disease', 'Has Disease']))

# Confusion Matrix Plot
plt.figure(figsize=(7, 5))
cm = confusion_matrix(y_test, best_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Disease', 'Has Disease'],
            yticklabels=['No Disease', 'Has Disease'],
            linewidths=1, linecolor='white',
            annot_kws={'size': 14, 'weight': 'bold'})
plt.title(f'Confusion Matrix — {best_model_name}', fontsize=13, fontweight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('plot11_confusion_matrix.png', dpi=150)
plt.show()
print("   ✓ Confusion Matrix saved")

# STEP 10: FEATURE IMPORTANCE (for Random Forest)

print("\n[STEP 10] Feature Importance (Random Forest)...")

feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importance = feature_importance.sort_values(ascending=True)

plt.figure(figsize=(9, 6))
colors_fi = ['#e74c3c' if v == feature_importance.max() else '#3498db'
             for v in feature_importance.values]
feature_importance.plot(kind='barh', color=colors_fi, edgecolor='black')
plt.title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('plot12_feature_importance.png', dpi=150)
plt.show()
print("   ✓ Feature Importance chart saved")

# FINAL SUMMARY

print("\n" + "=" * 55)
print("   PROJECT COMPLETE!")
print("=" * 55)
print(f"""
   SUMMARY:
   ─────────────────────────────────────────────────────
   Dataset     : {df.shape[0]} patients, {df.shape[1]} features
   Target      : cardio (0=No Disease, 1=Has Disease)
   
   MODEL ACCURACIES:
   ─────────────────────────────────────────────────────
   Logistic Regression : {results['Logistic Regression']*100:.2f}%
   KNN (k=5)           : {results['KNN']*100:.2f}%
   SVM (RBF kernel)    : {results['SVM']*100:.2f}%
   Decision Tree       : {results['Decision Tree']*100:.2f}%
   Random Forest       : {results['Random Forest']*100:.2f}%
   
   🏆 BEST MODEL: {best_model_name} ({best_acc*100:.2f}%)
   ─────────────────────────────────────────────────────
   
   SAVED PLOTS:
   ─────────────────────────────────────────────────────
   plot1_target_distribution.png
   plot2_age_distribution.png
   plot3_gender_vs_cardio.png
   plot4_cholesterol_vs_cardio.png
   plot5_bp_boxplot.png
   plot6_bmi_distribution.png
   plot7_lifestyle_factors.png
   plot8_pairplot.png
   plot9_correlation_matrix.png
   plot10_model_comparison.png
   plot11_confusion_matrix.png
   plot12_feature_importance.png
""")
