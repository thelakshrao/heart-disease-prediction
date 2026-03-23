🫀 Cardiovascular Disease Prediction

A Machine Learning project that predicts whether a person has heart disease or not — using their medical data like blood pressure, cholesterol, age, and lifestyle habits.


📖 What is this project about?
Heart disease (also called cardiovascular disease) is one of the biggest causes of death in India and around the world. In 2016 alone, more than 17.6 million people died from it.
The goal of this project is simple:

Can a computer learn from patient data and predict if someone has heart disease?

We use Machine Learning — a type of Artificial Intelligence — to answer this question. We train 5 different ML models on real patient data and find out which one is the most accurate.

📁 Dataset
The dataset contains medical records of thousands of patients. Each row is one patient.
ColumnWhat it meansageAge of the patient (converted from days to years)gender1 = Female, 2 = MaleheightHeight in cmweightWeight in kgap_hiSystolic blood pressure (the higher number, e.g. 120 in 120/80)ap_loDiastolic blood pressure (the lower number, e.g. 80 in 120/80)cholesterol1 = Normal, 2 = Above Normal, 3 = Well Above NormalglucGlucose level — 1 = Normal, 2 = Above Normal, 3 = Well Above NormalsmokeDoes the patient smoke? 0 = No, 1 = YesalcoDoes the patient drink alcohol? 0 = No, 1 = YesactiveIs the patient physically active? 0 = No, 1 = YescardioTARGET — Does the patient have heart disease? 0 = No, 1 = Yes

🛠️ Tools & Libraries Used
ToolPurposePython 3Main programming languagePandasLoading and cleaning the dataNumPyMathematical operationsMatplotlibDrawing graphs and chartsSeabornBeautiful statistical visualizationsScikit-learnBuilding and evaluating ML models

⚙️ How to Run This Project
Step 1 — Clone the repository
bashgit clone https://github.com/YOUR_USERNAME/heart-disease-prediction.git
cd heart-disease-prediction
Step 2 — Install required libraries
bashpip install pandas numpy matplotlib seaborn scikit-learn
Step 3 — Run the project
bashpython3 heart_disease_prediction.py
That's it! The code will automatically clean the data, create all the graphs, train the models, and show you the results.

🔄 Project Steps
Step 1 — Data Pre-Processing (Cleaning the Data)
Raw data is messy. Before training any model, we clean it:

Removed the id column (useless for prediction)
Converted age from days → years (e.g. 20000 days = ~54 years)
Removed rows with incorrect blood pressure values (negative or impossibly high)
Removed unrealistic height and weight values
Added a new BMI feature (Body Mass Index = weight / height²)

Step 2 — Data Analysis & Visualizations
We drew 12 graphs to understand the data better (see below).
Step 3 — Correlation Matrix
We checked which features are most related to heart disease.
Step 4 — Model Training
We trained 5 different Machine Learning models and compared their accuracy.
Step 5 — Best Model Selection
The best model was chosen and a detailed report was generated.

📊 Graphs & What They Tell Us
Graph 1 — Target Distribution
Show Image
This bar chart shows how many patients in the dataset have heart disease vs how many don't.

Ideally we want roughly equal numbers on both sides so the model doesn't get biased.


Graph 2 — Age Distribution by Disease
Show Image
This histogram shows the age spread of patients with and without heart disease.

Key insight: Older patients are much more likely to have heart disease. The red bars (disease) are shifted to the right (older ages).


Graph 3 — Gender vs Heart Disease
Show Image
This grouped bar chart compares heart disease rates between males and females.

Key insight: Shows if one gender is more at risk than the other in this dataset.


Graph 4 — Cholesterol Level vs Heart Disease
Show Image
Cholesterol levels are grouped into 3 categories: Normal, Above Normal, Well Above Normal.

Key insight: Patients with higher cholesterol are far more likely to have heart disease.


Graph 5 — Blood Pressure Boxplot
Show Image
A boxplot showing the range of systolic blood pressure for patients with and without heart disease.

Key insight: Patients with heart disease tend to have significantly higher blood pressure.


Graph 6 — BMI Distribution by Disease
Show Image
BMI (Body Mass Index) is a measure of body fat based on height and weight.

Key insight: Patients with heart disease tend to have a higher BMI (overweight/obese range).


Graph 7 — Lifestyle Factors (Smoking, Alcohol, Activity)
Show Image
Three side-by-side charts showing how smoking, alcohol use, and physical activity relate to heart disease.

Key insight: Physically active patients have lower rates of heart disease. Smoking and alcohol show some correlation too.


Graph 8 — Pairplot of Key Features
Show Image
This grid of mini-charts shows the relationship between every pair of important features.

Key insight: Age, BMI, and blood pressure together show clear separation between disease and no-disease patients.


Graph 9 — Correlation Matrix
Show Image
A heatmap where each box shows how strongly two features are related to each other.

Values close to +1 = strong positive relationship
Values close to -1 = strong negative relationship
Values close to 0 = no relationship
Key insight: age, ap_hi, ap_lo, cholesterol, and bmi are most correlated with having heart disease.


Graph 10 — Model Accuracy Comparison
Show Image
A bar chart comparing the accuracy of all 5 Machine Learning models side by side.

The best model is highlighted in red.


Graph 11 — Confusion Matrix (Best Model)
Show Image
A confusion matrix shows exactly where the model got things right and wrong:

Top-left: Correctly predicted NO disease ✅
Bottom-right: Correctly predicted HAS disease ✅
Top-right: Said "has disease" but patient was fine ❌
Bottom-left: Said "no disease" but patient actually had it ❌


Graph 12 — Feature Importance (Random Forest)
Show Image
This horizontal bar chart shows which features matter the most when making predictions.

Key insight: Age, systolic blood pressure (ap_hi), diastolic pressure (ap_lo), BMI, and cholesterol are the top predictors of heart disease.


🤖 Machine Learning Models Used
We trained and tested 5 different models:
1. Logistic Regression (LR)

The simplest model — it draws a straight line to separate the two groups.
Fast and easy to understand.
Good as a baseline to compare against other models.

2. K-Nearest Neighbor (KNN)

Classifies a patient by looking at the 5 closest similar patients in the data.
Simple idea: if your 5 nearest neighbors all have heart disease, you probably do too.
Works well but can be slow on very large datasets.

3. Support Vector Machine (SVM)

Finds the best possible boundary between the two groups (disease vs no disease).
Works well even with complex data.
Uses a mathematical trick called the "RBF kernel" to handle non-linear patterns.

4. Decision Tree (DT)

Works like a flowchart — asks yes/no questions step by step to reach a conclusion.
Example: "Is age > 50? → Is blood pressure > 130? → ..."
Very easy to visualize and explain to non-technical people.

5. Random Forest (RF)

Builds 100 decision trees and takes a majority vote from all of them.
Much stronger and more accurate than a single decision tree.
Usually the best performer in medical datasets — and it was here too!


📈 Results
ModelAccuracyLogistic Regression~71%K-Nearest Neighbor~69%Support Vector Machine~72%Decision Tree~65%Random Forest~73%

🏆 Random Forest gave the best accuracy and was selected as the final model.


💡 Key Findings
After analyzing all the data and graphs, here are the most important things we found:

Age is the strongest predictor — older patients are much more likely to have heart disease.
High blood pressure (both systolic and diastolic) is strongly linked to heart disease.
High cholesterol significantly increases the risk.
Higher BMI (overweight/obese patients) are at higher risk.
Physical activity helps — active patients have lower disease rates.
Smoking and alcohol showed some effect but were less impactful than the above factors.


📂 Project Structure
heart_disease_project/
│
├── cardio_train.csv                  ← Dataset
├── heart_disease_prediction.py       ← Main Python code
├── README.md                         ← This file
│
├── plot1_target_distribution.png
├── plot2_age_distribution.png
├── plot3_gender_vs_cardio.png
├── plot4_cholesterol_vs_cardio.png
├── plot5_bp_boxplot.png
├── plot6_bmi_distribution.png
├── plot7_lifestyle_factors.png
├── plot8_pairplot.png
├── plot9_correlation_matrix.png
├── plot10_model_comparison.png
├── plot11_confusion_matrix.png
└── plot12_feature_importance.png

👨‍💻 Author
Made as part of an AI Class Project.

📜 License
This project is open source and free to use for educational purposes.
