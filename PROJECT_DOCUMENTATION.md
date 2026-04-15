# 🏋️ FitAI – Fitness Recommendation System
### *Machine Learning Project Documentation*
---

## 📌 Project Title
**Fitness Recommendation System using Machine Learning with Streamlit UI**

| Detail | Value |
|---|---|
| **Project Type** | Machine Learning + Web Application |
| **Platform** | Python + Streamlit |
| **ML Models** | Decision Tree Classifier + Linear Regression |
| **Dataset** | exercise_dataset.csv (3,864 records) |
| **Organization** | Infosys Internship Project |
| **Language** | Python 3.12 |

---

## 📖 Introduction

The **FitAI Fitness Recommendation System** is a data-driven web application that provides personalized workout and diet suggestions based on a user's health data. The system leverages two machine learning models trained on real exercise and fitness metrics data:

1. A **Decision Tree Classifier** to categorize the user's fitness level (Normal / Overweight / Underweight)
2. A **Linear Regression model** to estimate calorie burn per exercise session

Based on these ML predictions, the system generates tailored workout plans, weekly schedules, and diet recommendations.

---

## 🎯 Objectives

- ✅ Analyze user health parameters (age, weight, height, gender, heart rate)
- ✅ Calculate BMI and determine fitness category using ML
- ✅ Predict calorie burn per session using Linear Regression
- ✅ Recommend personalized workout plans based on fitness category
- ✅ Recommend personalized diet plans based on fitness category
- ✅ Show dataset benchmarks comparing the user to similar profiles
- ✅ Build a premium, user-friendly Streamlit interface

---

## 📊 Dataset Description

### File: `exercise_dataset.csv`

| Column | Type | Description | Used In |
|---|---|---|---|
| `ID` | int | Row identifier | — |
| `Age` | int | Age in years | ✅ Both Models |
| `Gender` | str | Male / Female | ✅ Both Models (encoded) |
| `Actual Weight` | float | Current weight in kg | ✅ Decision Tree |
| `Dream Weight` | float | Goal weight in kg | ✅ Both Models |
| `BMI` | float | Body Mass Index | ✅ Both Models |
| `Calories Burn` | float | Calories burned in session | ✅ LR Target |
| `Duration` | int | Exercise duration (minutes) | ✅ Linear Regression |
| `Heart Rate` | int | Avg heart rate during session | ✅ Linear Regression |
| `Exercise Intensity` | int | Scale 1–10 | ✅ Both Models |
| `Exercise` | str | Exercise type label (1–10) | — |
| `Weather Conditions` | str | Rainy / Cloudy / Sunny | — |

### Dataset Statistics

| Metric | Value |
|---|---|
| **Total Records** | 3,864 |
| **Male Participants** | 1,900 |
| **Female Participants** | 1,964 |
| **BMI Range** | ~18.5 – ~45+ |
| **Age Range** | ~15 – ~75 years |
| **Duration Range** | 5 – 120 minutes |
| **Heart Rate Range** | 60 – 200 bpm |
| **Exercise Intensity** | 1 – 10 |
| **Avg Calories Burned** | ~295–305 kcal |

---

## ⚙️ System Requirements

### 🖥️ Hardware
| Component | Minimum |
|---|---|
| RAM | 4 GB |
| Processor | Any modern dual-core CPU |
| Storage | ~200 MB (for libraries + dataset) |

### 💻 Software
| Tool | Version |
|---|---|
| Python | 3.8 or above (tested on 3.12) |
| pip | 24+ |
| Streamlit | 1.x |
| scikit-learn | 1.x |
| pandas | 2.x |
| numpy | 2.x |

---

## 🔄 Methodology

### Step 1: Data Loading & Preprocessing

```python
df = pd.read_csv("exercise_dataset.csv")

# Drop missing values (none in this dataset)
df = df.dropna().reset_index(drop=True)

# Encode Gender: Female=0, Male=1
le_gender = LabelEncoder()
df["Gender_enc"] = le_gender.fit_transform(df["Gender"])

# Derived Feature: Weight-to-Dream Gap
df["Weight_Diff"] = df["Actual Weight"] - df["Dream Weight"]

# Derive BMI Category (classification target)
def classify_bmi(bmi):
    if bmi < 18.5:  return "Underweight"
    elif bmi < 25:  return "Normal"
    else:           return "Overweight"

df["BMI_Category"] = df["BMI"].apply(classify_bmi)
```

---

### Step 2: BMI Calculation

```
BMI = Weight (kg) / Height² (m²)
```

| BMI Range | Category |
|---|---|
| < 18.5 | 🟡 Underweight |
| 18.5 – 24.9 | 🟢 Normal |
| 25.0 – 29.9 | 🔴 Overweight |
| ≥ 30.0 | 🔴 Obese |

---

### Step 3: Model 1 – Decision Tree Classifier

#### Purpose
Classify users into BMI fitness categories: **Normal** or **Overweight** (Underweight if applicable).

#### Input Features
```
Age, BMI, Actual Weight, Dream Weight, Weight_Diff, Gender_enc, Exercise Intensity
```

#### Target Variable
```
BMI_Category  →  'Normal' | 'Overweight' | 'Underweight'
```

#### Configuration
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X_clf = df[["Age","BMI","Actual Weight","Dream Weight","Weight_Diff","Gender_enc","Exercise Intensity"]]
y_clf = df["BMI_Category"]

X_train, X_test, y_train, y_test = train_test_split(
    X_clf, y_clf, test_size=0.20, random_state=42, stratify=y_clf
)

clf = DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, random_state=42)
clf.fit(X_train, y_train)
```

#### Results
| Metric | Value |
|---|---|
| **Test Accuracy** | 100% |
| **Train/Test Split** | 80% / 20% |
| **Test Set Size** | ~773 samples |

> **Note:** High accuracy is expected because `BMI_Category` is deterministically derived from the `BMI` column (which is present in training data). In a production scenario, BMI would be calculated from user input alone — which the app correctly does.

---

### Step 4: Model 2 – Linear Regression (Calories Burn)

#### Purpose
Predict the estimated number of **calories burned** in a given exercise session.

#### Input Features
```
Age, Duration, Heart Rate, BMI, Exercise Intensity, Weight_Diff, Gender_enc
```

#### Target Variable
```
Calories Burn  →  Continuous float (kcal)
```

#### Configuration
```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

X_reg = df[["Age","Duration","Heart Rate","BMI","Exercise Intensity","Weight_Diff","Gender_enc"]]
y_reg = df["Calories Burn"]

X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.20, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

reg = LinearRegression()
reg.fit(X_train_s, y_train)
```

#### Results
| Metric | Value |
|---|---|
| **MAE** | ~100 kcal |
| **R² Score** | ~0.00 |
| **Train/Test Split** | 80% / 20% |
| **Feature Scaling** | StandardScaler (z-score) |

> **Note:** The low R² indicates that `Calories Burn` in this dataset is approximately uniformly distributed (like a random draw) independent of the features. The LR model still provides a data-driven estimate using the intercept (~299 kcal) plus small feature adjustments. The Dataset Benchmark section supplements this with real peer-group averages.

---

### Step 5: Recommendation Engine

Based on the **predicted BMI category** from the Decision Tree:

#### 🔴 Overweight Category
| Component | Recommendations |
|---|---|
| **Workout** | Running/Jogging (30–45 min), Cycling, Swimming, HIIT, Yoga cool-down |
| **Diet** | Low-calorie high-fibre meals, fruits/vegetables, no sugar/fried, 3-4L water |
| **Calorie Goal** | Deficit of 300–500 kcal/day |

#### 🟡 Underweight Category
| Component | Recommendations |
|---|---|
| **Workout** | Strength/Weight Training (3×/wk), Resistance Bands, Bodyweight, Pilates |
| **Diet** | High-protein (eggs, chicken, lentils), full-fat dairy, complex carbs, calorie-dense snacks |
| **Calorie Goal** | Surplus of 300–500 kcal/day |

#### 🟢 Normal Category
| Component | Recommendations |
|---|---|
| **Workout** | Yoga & Mindfulness, moderate cycling/swimming, jogging, recreational sports |
| **Diet** | Balanced (40% carbs, 30% protein, 30% fat), seasonal fruits, lean proteins, whole grains |
| **Calorie Goal** | Maintenance calories (TDEE balance) |

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    USER INPUT (Sidebar)                   │
│  Age · Gender · Weight · Dream Weight · Height            │
│  Duration · Heart Rate · Exercise Intensity · Goal        │
└───────────────────────┬──────────────────────────────────┘
                        │
                        ▼
           ┌────────────────────────┐
           │   DATA PREPROCESSING   │
           │  • BMI Calculation     │
           │  • Gender Encoding     │
           │  • Weight Difference   │
           │  • Feature Scaling     │
           └────────────┬───────────┘
                        │
           ┌────────────┴───────────────────────────┐
           │                                        │
           ▼                                        ▼
 ┌──────────────────────┐              ┌────────────────────────┐
 │ Decision Tree (DT)   │              │  Linear Regression (LR)│
 │ Trained on 3,091 rows│              │  Trained on 3,091 rows │
 │ 7 input features     │              │  7 input features      │
 │ Output: BMI Category │              │  Output: Calories(kcal)│
 └──────────┬───────────┘              └────────────┬───────────┘
            │                                       │
            ▼                                       ▼
 ┌──────────────────────────────────────────────────────────┐
 │                  RECOMMENDATION ENGINE                    │
 │  Based on predicted BMI Category:                        │
 │  • Overweight  → Cardio plan + low-calorie diet          │
 │  • Normal      → Maintenance plan + balanced diet        │
 │  • Underweight → Strength plan + high-protein diet       │
 │  + Goal-specific tips (Lose / Gain / Maintain / Endure)  │
 └──────────────────────────┬───────────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │   STREAMLIT UI OUTPUT   │
              │  • Health Summary Cards │
              │  • Workout Plan         │
              │  • Diet Plan            │
              │  • Weekly Schedule      │
              │  • Dataset Benchmarks   │
              └─────────────────────────┘
```

---

## 🖥️ Streamlit UI – Features

### Sidebar Input Fields
| Field | Type | Range |
|---|---|---|
| Age | Slider | 10 – 80 years |
| Gender | Radio | Male / Female |
| Current Weight | Number Input | 30 – 200 kg |
| Dream Weight | Number Input | 30 – 200 kg |
| Height | Number Input | 100 – 220 cm |
| Exercise Duration | Slider | 5 – 120 min |
| Heart Rate | Slider | 60 – 200 bpm |
| Exercise Intensity | Slider | 1 – 10 |
| Fitness Goal | Dropdown | 4 options |

### Output Tabs

| Tab | Contents |
|---|---|
| 🎯 My Fitness Plan | Live BMI preview → On predict: health summary cards, BMI scale, workout plan, diet plan, weekly schedule, dataset benchmarks for similar profiles |
| 📊 Dataset Insights | Dataset overview metrics, 6 bar charts (age, intensity, BMI dist, gender duration, HR by BMI, weight comparison), raw data table, descriptive statistics |
| 🤖 Model Performance | Accuracy / MAE / R² metrics, per-class classification report, confusion matrix, feature importance chart, coefficient table |
| 📋 How It Works | System architecture diagram, dataset column descriptions, BMI formula, tech stack |

---

## 📌 Example Output

**Input:**
```
Age:        25
Gender:     Male
Weight:     75 kg
Dream Wt:   65 kg
Height:     170 cm
Duration:   30 min
Heart Rate: 140 bpm
Intensity:  5
Goal:       Lose Weight
```

**Output:**
```
BMI:                  25.95  →  Overweight (Decision Tree: Overweight)
Est. Calories Burned: ~290 kcal  (Linear Regression)
vs Dream Weight:      +10.0 kg   (needs to lose 10 kg)
Ideal Weight Range:   53.5–72.0 kg

Workout Plan: Running, Cycling, HIIT, Swimming
Diet Plan:    Low-calorie, high-fibre, fruits, vegetables

Weekly Schedule:
  Mon → Running 30 min + Core
  Tue → Cycling 40 min
  Wed → Rest / Yoga
  Thu → HIIT 25 min
  Fri → Swimming 30 min
  Sat → Yoga & flexibility
  Sun → Rest & Recovery
```

---

## 📂 Project File Structure

```
fitness/
│
├── app.py                   # Main Streamlit application
├── exercise_dataset.csv     # Training dataset (3,864 rows × 12 cols)
└── PROJECT_DOCUMENTATION.md # This file – complete project documentation
```

---

## 🚀 How to Run the Project

### Step 1: Install Dependencies
```bash
pip install streamlit pandas numpy scikit-learn
```

### Step 2: Navigate to Project Folder
```bash
cd "c:\Users\rautr\OneDrive\Desktop\fitness"
```

### Step 3: Run the App
```bash
python -m streamlit run app.py
```

### Step 4: Open Browser
```
http://localhost:8501
```

---

## 📚 Libraries Used

| Library | Version | Purpose |
|---|---|---|
| `streamlit` | 1.x | Web application UI framework |
| `pandas` | 2.x | Data loading and manipulation |
| `numpy` | 2.x | Numerical computations |
| `scikit-learn` | 1.x | Decision Tree, Linear Regression, metrics |
| `sklearn.preprocessing` | — | LabelEncoder, StandardScaler |
| `sklearn.model_selection` | — | train_test_split |
| `sklearn.metrics` | — | accuracy_score, MAE, R², confusion_matrix |

---

## ⚠️ Challenges & Limitations

| Challenge | Description |
|---|---|
| **Dataset Calories** | `Calories Burn` is nearly uniformly distributed in the dataset, making Linear Regression R² close to 0 — supplemented with dataset peer benchmarks |
| **No Underweight Records** | The dataset contains no BMI < 18.5 entries; the Underweight recommendation is rule-based |
| **Exercise Labels** | Exercise types are labeled Exercise 1–10 (not real names), so workout names are curated |
| **Generalized Plans** | Recommendations are category-based, not individualized for medical conditions |
| **No External API** | No wearable device integration; all inputs are manual |

---

## 🚀 Future Scope

- 📡 Integration with wearable devices (Fitbit, Apple Watch API)
- 📲 Mobile application development (React Native / Flutter)
- 🤖 AI-powered personalized diet planning using NLP
- 📊 Real-time fitness tracking dashboard
- 🧬 Integration of more health metrics (sleep, stress, VO2 max)
- 🗺️ Multi-language support
- 🔐 User login and fitness history tracking

---

## 🧾 Conclusion

The **FitAI Fitness Recommendation System** successfully demonstrates:

1. **End-to-end ML pipeline** – from raw CSV data to trained models serving live predictions
2. **Two ML models** working together – Decision Tree for classification and Linear Regression for regression
3. **Practical UI** – Streamlit-based interface with sidebar inputs, metric cards, charts, and data tables
4. **Data-driven benchmarking** – comparing user profiles against real dataset records
5. **Comprehensive documentation** – covering methodology, architecture, results, and how-to

> 💡 **Viva Line:** *"This project uses a Decision Tree Classifier to predict fitness categories from BMI and user health data, and a Linear Regression model to estimate calories burned per session — together driving a personalized fitness recommendation engine built with Streamlit."*

---

## 📌 Keywords

`Machine Learning` · `Fitness` · `BMI` · `Recommendation System` · `Decision Tree` · `Linear Regression` · `Streamlit` · `scikit-learn` · `Python` · `Exercise Dataset` · `Calorie Prediction` · `Health Analytics`

---

*Generated for Infosys Internship Project — FitAI Fitness Recommendation System*
