"""
=============================================================
  FitAI – Fitness Recommendation System
  Machine Learning with Streamlit UI
  Dataset: Exercise and Fitness Metrics (exercise_dataset.csv)
  Models:
      1. Decision Tree Classifier  → BMI Category prediction
      2. Linear Regression         → Calories Burn estimation
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, mean_absolute_error,
                              r2_score, classification_report,
                              confusion_matrix)

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FitAI – Fitness Recommendation System",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Premium Dark Theme CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .stApp { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); min-height: 100vh; }

  /* Hero */
  .hero {
    background: linear-gradient(135deg, rgba(99,102,241,.30) 0%, rgba(168,85,247,.30) 100%);
    border: 1px solid rgba(168,85,247,.45);
    border-radius: 22px; padding: 2.5rem 2rem;
    text-align: center; margin-bottom: 2rem;
    backdrop-filter: blur(12px);
  }
  .hero h1 {
    font-size: 2.7rem; font-weight: 800; margin: 0;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .hero p  { color: #cbd5e1; font-size: 1.1rem; margin-top: .6rem; }
  .hero .pill {
    display: inline-block; margin: .4rem .25rem 0;
    padding: .25rem .85rem; border-radius: 999px; font-size: .8rem; font-weight: 600;
    background: rgba(167,139,250,.2); color: #a78bfa; border: 1px solid rgba(167,139,250,.4);
  }

  /* Section title */
  .section-title {
    font-size: 1.2rem; font-weight: 700; color: #a78bfa;
    border-left: 4px solid #a78bfa; padding-left: .8rem; margin-bottom: .9rem;
  }

  /* Metric cards */
  .metric-card {
    background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.10);
    border-radius: 16px; padding: 1.3rem 1rem; text-align: center;
    transition: transform .2s ease, border-color .2s ease;
  }
  .metric-card:hover { transform: translateY(-4px); border-color: rgba(167,139,250,.6); }
  .metric-value { font-size: 2rem; font-weight: 800; }
  .metric-label { color: #94a3b8; font-size: .82rem; margin-top: .3rem; }

  /* Badges */
  .badge { display: inline-block; padding: .4rem 1.1rem; border-radius: 999px; font-weight: 700; font-size: .95rem; }
  .badge-normal { background: rgba(52,211,153,.15); color: #34d399; border: 1px solid #34d399; }
  .badge-over   { background: rgba(248,113,113,.15); color: #f87171; border: 1px solid #f87171; }
  .badge-under  { background: rgba(251,191,36,.15);  color: #fbbf24; border: 1px solid #fbbf24; }

  /* Rec boxes */
  .rec-box {
    background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.11);
    border-radius: 14px; padding: 1.2rem 1.4rem; margin-bottom: .8rem;
  }
  .rec-box h4 { color: #60a5fa; margin: 0 0 .5rem; font-size: 1rem; }
  .rec-box ul { margin: 0; padding-left: 1.2rem; color: #e2e8f0; }
  .rec-box li { margin-bottom: .35rem; }

  /* Alert overrides */
  .stAlert { border-radius: 12px; }

  /* Progress bar */
  .stProgress > div > div { background: linear-gradient(90deg, #a78bfa, #60a5fa); border-radius: 999px; }

  /* Sidebar */
  section[data-testid="stSidebar"] { background: rgba(12,10,35,.92); border-right: 1px solid rgba(255,255,255,.08); }

  /* Divider */
  hr { border-color: rgba(255,255,255,.09); }

  /* Table */
  .stDataFrame { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Load Dataset & Train Models (Cached)
# ─────────────────────────────────────────────
@st.cache_data
def load_and_train():
    # ── 1. Load raw CSV ───────────────────────────────────────────────────────
    df = pd.read_csv("exercise_dataset.csv")

    # ── 2. Data Preprocessing ────────────────────────────────────────────────
    df = df.dropna()                          # Drop any missing rows
    df = df.reset_index(drop=True)

    # Encode Gender: Female=0, Male=1
    le_gender = LabelEncoder()
    df["Gender_enc"] = le_gender.fit_transform(df["Gender"])

    # ── 3. Derived Feature: Weight-to-Dream Difference ────────────────────────
    df["Weight_Diff"] = df["Actual Weight"] - df["Dream Weight"]

    # ── 4. BMI Category Labels (target for classification) ───────────────────
    # Dataset has NO underweight entries; we keep three classes and handle it
    def classify_bmi(bmi):
        if bmi < 18.5:
            return "Underweight"
        elif bmi < 25.0:
            return "Normal"
        else:
            return "Overweight"

    df["BMI_Category"] = df["BMI"].apply(classify_bmi)

    # ── 5. Model 1: Decision Tree Classifier ─────────────────────────────────
    # Features: Age, BMI, Actual Weight, Dream Weight, Weight Diff, Gender, Intensity
    clf_features = ["Age", "BMI", "Actual Weight", "Dream Weight",
                    "Weight_Diff", "Gender_enc", "Exercise Intensity"]
    X_clf = df[clf_features]
    y_clf = df["BMI_Category"]

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_clf, y_clf, test_size=0.20, random_state=42, stratify=y_clf
    )
    clf = DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, random_state=42)
    clf.fit(X_train_c, y_train_c)
    y_pred_c = clf.predict(X_test_c)

    clf_acc     = accuracy_score(y_test_c, y_pred_c)
    clf_report  = classification_report(y_test_c, y_pred_c, output_dict=True)
    clf_cm      = confusion_matrix(y_test_c, y_pred_c, labels=clf.classes_)
    clf_classes = list(clf.classes_)

    # ── 6. Model 2: Linear Regression – Calories Burn ────────────────────────
    # Features: Age, Duration, Heart Rate, BMI, Exercise Intensity, Weight Diff, Gender
    reg_features = ["Age", "Duration", "Heart Rate", "BMI",
                    "Exercise Intensity", "Weight_Diff", "Gender_enc"]
    X_reg = df[reg_features]
    y_reg = df["Calories Burn"]

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg, test_size=0.20, random_state=42
    )

    scaler = StandardScaler()
    X_train_rs = scaler.fit_transform(X_train_r)
    X_test_rs  = scaler.transform(X_test_r)

    reg = LinearRegression()
    reg.fit(X_train_rs, y_train_r)
    y_pred_r = reg.predict(X_test_rs)

    reg_mae  = mean_absolute_error(y_test_r, y_pred_r)
    reg_rmse = np.sqrt(np.mean((y_test_r - y_pred_r) ** 2))
    reg_r2   = r2_score(y_test_r, y_pred_r)

    # Coefficients mapped to feature names (for documentation tab)
    reg_coefs = dict(zip(reg_features, reg.coef_.round(4)))

    return (
        df, clf, clf_features, clf_acc, clf_report, clf_cm, clf_classes,
        reg, reg_features, scaler, reg_mae, reg_rmse, reg_r2, reg_coefs
    )


(df, clf, clf_features, clf_acc, clf_report, clf_cm, clf_classes,
 reg, reg_features, scaler, reg_mae, reg_rmse, reg_r2, reg_coefs) = load_and_train()


# ─────────────────────────────────────────────
# Recommendation Content
# ─────────────────────────────────────────────
RECS = {
    "Overweight": {
        "icon": "🔴",
        "color": "badge-over",
        "calorie_goal": "Calorie DEFICIT of 300–500 kcal/day",
        "workout": [
            "🏃 Running / Jogging – 30–45 min daily",
            "🚴 Cycling – moderate intensity, 40 min",
            "🏊 Swimming – full‑body cardio, 3× / week",
            "⚡ HIIT – High Intensity Interval Training",
            "🧘 Yoga + Stretching for cool‑down",
        ],
        "diet": [
            "🥗 Low‑calorie, high‑fibre meals",
            "🍎 Plenty of fruits & leafy vegetables",
            "🚫 Avoid processed sugar and fried food",
            "💧 Drink 3–4 litres of water daily",
            "🥜 Healthy fats: avocado, nuts (in moderation)",
        ],
        "weekly": {
            "Monday":    "🏃 Running 30 min + Core circuit",
            "Tuesday":   "🚴 Cycling 40 min (moderate)",
            "Wednesday": "😴 Rest / Light yoga & stretching",
            "Thursday":  "⚡ HIIT session – 25 min",
            "Friday":    "🏊 Swimming 30 min",
            "Saturday":  "🧘 Yoga & full‑body flexibility",
            "Sunday":    "😴 Rest & Active Recovery (walk)",
        },
    },
    "Underweight": {
        "icon": "🟡",
        "color": "badge-under",
        "calorie_goal": "Calorie SURPLUS of 300–500 kcal/day",
        "workout": [
            "🏋️ Strength / Weight Training – 3× / week",
            "🤸 Resistance Band exercises",
            "🧗 Bodyweight: push‑ups, pull‑ups, squats",
            "🚶 Brisk Walking – light cardio only",
            "🧘 Pilates for core strength",
        ],
        "diet": [
            "🥩 High‑protein: eggs, chicken, lentils",
            "🥛 Full‑fat dairy: milk, cottage cheese",
            "🍚 Complex carbs: brown rice, oats, sweet potato",
            "🥜 Calorie‑dense snacks: peanut butter, nuts",
            "🍌 Protein smoothies with banana & oats",
        ],
        "weekly": {
            "Monday":    "🏋️ Upper Body Strength training",
            "Tuesday":   "🚶 Brisk Walk 30 min + Core",
            "Wednesday": "🏋️ Lower Body Strength training",
            "Thursday":  "😴 Rest / Light stretching",
            "Friday":    "🏋️ Full Body compound lifts",
            "Saturday":  "🧘 Pilates / Flexibility work",
            "Sunday":    "😴 Rest & Recovery",
        },
    },
    "Normal": {
        "icon": "🟢",
        "color": "badge-normal",
        "calorie_goal": "Maintenance Calories (TDEE balance)",
        "workout": [
            "🧘 Yoga & Mindfulness – 3× / week",
            "🚴 Moderate Cycling or Swimming",
            "🏃 Light Jogging – 25–30 min, 3× / week",
            "🏸 Recreational sports: badminton, tennis",
            "🤸 Functional Fitness / bodyweight circuits",
        ],
        "diet": [
            "🥗 Balanced: 40% carbs, 30% protein, 30% fats",
            "🍓 Seasonal fruits & colourful vegetables",
            "🐟 Lean protein: fish, eggs, legumes",
            "🌾 Whole grains over refined carbs",
            "💧 Stay hydrated, limit sugary drinks",
        ],
        "weekly": {
            "Monday":    "🏃 Jogging 25 min",
            "Tuesday":   "🏋️ Bodyweight strength training",
            "Wednesday": "🧘 Yoga / Mindfulness",
            "Thursday":  "🚴 Cycling 30 min",
            "Friday":    "🏸 Recreational sport",
            "Saturday":  "🤸 Functional fitness circuit",
            "Sunday":    "😴 Rest & Recovery",
        },
    },
}

GOAL_TIPS = {
    "Lose Weight":       "Focus on a consistent calorie deficit. Combine cardio and strength training, and prioritise sleep & stress management.",
    "Gain Muscle":       "Prioritise progressive overload in strength training. Eat in a calorie surplus with 1.6–2.2 g protein per kg body weight.",
    "Maintain Fitness":  "Keep a mixed routine of cardio and strength. Match calorie intake to your TDEE and get 7–9 hours of quality sleep.",
    "Improve Endurance": "Gradually increase weekly workout duration by ~10%. Fuel with complex carbs before sessions and recover actively.",
}


# ─────────────────────────────────────────────
# Prediction Helpers
# ─────────────────────────────────────────────
def compute_bmi(weight_kg: float, height_cm: float) -> float:
    h = height_cm / 100.0
    return weight_kg / (h * h)


def classify_bmi_rule(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25.0:
        return "Normal"
    else:
        return "Overweight"


def ml_predict_category(age, bmi, actual_wt, dream_wt, gender, intensity):
    """Run Decision Tree prediction using the trained model."""
    gender_enc = 1 if gender == "Male" else 0
    wt_diff = actual_wt - dream_wt
    X = np.array([[age, bmi, actual_wt, dream_wt, wt_diff, gender_enc, intensity]])
    return clf.predict(X)[0]


def ml_predict_calories(age, duration, heart_rate, bmi, intensity, actual_wt, dream_wt, gender):
    """Run Linear Regression prediction using the trained model."""
    gender_enc = 1 if gender == "Male" else 0
    wt_diff = actual_wt - dream_wt
    X = np.array([[age, duration, heart_rate, bmi, intensity, wt_diff, gender_enc]])
    X_scaled = scaler.transform(X)
    cal = reg.predict(X_scaled)[0]
    # Fallback: clamp to a sensible range
    cal = float(np.clip(cal, 50, 1500))
    return cal


# ─────────────────────────────────────────────
# Sidebar – User Inputs
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👤 Your Profile")
    st.markdown("---")

    age    = st.slider("🎂 Age (years)", 10, 80, 28)
    gender = st.radio("⚡ Gender", ["Male", "Female"], horizontal=True)

    st.markdown("---")
    weight_kg = st.number_input("⚖️ Current Weight (kg)", 30.0, 200.0, 75.0, step=0.5)
    dream_wt  = st.number_input("🎯 Dream / Goal Weight (kg)", 30.0, 200.0, 65.0, step=0.5)
    height_cm = st.number_input("📏 Height (cm)", 100.0, 220.0, 170.0, step=0.5)

    # Live BMI
    bmi_live = compute_bmi(weight_kg, height_cm)
    st.markdown(f"**📐 BMI:** `{bmi_live:.2f}` — *{classify_bmi_rule(bmi_live)}*")

    st.markdown("---")
    duration   = st.slider("⏱️ Exercise Duration (min)", 5, 120, 30)
    heart_rate = st.slider("❤️ Heart Rate (bpm)", 60, 200, 130)
    intensity  = st.slider("🔥 Exercise Intensity (1–10)", 1, 10, 5)

    st.markdown("---")
    fitness_goal = st.selectbox(
        "🎯 Your Fitness Goal",
        ["Lose Weight", "Gain Muscle", "Maintain Fitness", "Improve Endurance"]
    )

    generate = st.button("🚀  Generate My Fitness Plan", use_container_width=True)


# ─────────────────────────────────────────────
# Hero Banner
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🏋️ FitAI – Fitness Recommendation System</h1>
  <p>Personalized workout &amp; diet plans powered by Machine Learning</p>
  <span class="pill">Decision Tree Classifier</span>
  <span class="pill">Linear Regression</span>
  <span class="pill">3,864 Real Records</span>
  <span class="pill">Streamlit UI</span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 My Fitness Plan",
    "📊 Dataset Insights",
    "🤖 Model Performance",
    "📋 How It Works",
])


# ════════════════════════════════════════════════
# TAB 1 – MY FITNESS PLAN
# ════════════════════════════════════════════════
with tab1:
    if not generate:
        st.info("👈 Fill in your details in the **left sidebar** and click **Generate My Fitness Plan** to get started.")

        # Live BMI preview cards
        c1, c2, c3 = st.columns(3)
        live_cat   = classify_bmi_rule(bmi_live)
        h_m        = height_cm / 100
        ideal_low  = 18.5 * h_m ** 2
        ideal_high = 24.9 * h_m ** 2

        with c1:
            st.markdown(f"""<div class="metric-card">
              <div class="metric-value" style="color:#60a5fa;">{bmi_live:.1f}</div>
              <div class="metric-label">Your BMI (live preview)</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            badge = RECS[live_cat]["color"]
            icon  = RECS[live_cat]["icon"]
            st.markdown(f"""<div class="metric-card">
              <div class="metric-value">{icon}</div>
              <div class="metric-label"><span class="badge {badge}">{live_cat}</span></div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card">
              <div class="metric-value" style="color:#34d399;font-size:1.1rem;">{ideal_low:.1f}–{ideal_high:.1f} kg</div>
              <div class="metric-label">Ideal Weight for Your Height</div>
            </div>""", unsafe_allow_html=True)

    else:
        # ── Run both ML models ─────────────────────────────────────────
        ml_category = ml_predict_category(age, bmi_live, weight_kg, dream_wt, gender, intensity)
        ml_calories = ml_predict_calories(age, duration, heart_rate, bmi_live,
                                          intensity, weight_kg, dream_wt, gender)
        rec         = RECS[ml_category]
        h_m         = height_cm / 100
        ideal_low   = 18.5 * h_m ** 2
        ideal_high  = 24.9 * h_m ** 2
        wt_diff     = weight_kg - dream_wt

        # ── Success banner ─────────────────────────────────────────────
        st.success(f"✅ Analysis complete for **{gender}, Age {age}** — here is your personalised plan!")

        # ── 5-column summary row ───────────────────────────────────────
        st.markdown('<div class="section-title">📋 Health Summary</div>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            st.markdown(f"""<div class="metric-card">
              <div class="metric-value" style="color:#60a5fa;">{bmi_live:.1f}</div>
              <div class="metric-label">Your BMI</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            badge = rec["color"]
            icon  = rec["icon"]
            st.markdown(f"""<div class="metric-card">
              <div class="metric-value">{icon}</div>
              <div class="metric-label">
                <span class="badge {badge}">{ml_category}</span><br>
                <small>(ML Predicted)</small>
              </div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card">
              <div class="metric-value" style="color:#f87171;">{ml_calories:.0f}</div>
              <div class="metric-label">Est. Calories Burned<br><small>(Linear Regression)</small></div>
            </div>""", unsafe_allow_html=True)
        with c4:
            diff_color = "#f87171" if wt_diff > 0 else "#34d399"
            diff_sign  = "+" if wt_diff >= 0 else ""
            st.markdown(f"""<div class="metric-card">
              <div class="metric-value" style="color:{diff_color};">{diff_sign}{wt_diff:.1f} kg</div>
              <div class="metric-label">vs Dream Weight</div>
            </div>""", unsafe_allow_html=True)
        with c5:
            st.markdown(f"""<div class="metric-card">
              <div class="metric-value" style="color:#34d399;font-size:1.1rem;">{ideal_low:.1f}–{ideal_high:.1f}</div>
              <div class="metric-label">Ideal Weight Range (kg)</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── BMI Scale ─────────────────────────────────────────────────
        st.markdown('<div class="section-title">📏 BMI Scale Position</div>', unsafe_allow_html=True)
        bmi_pct = float(np.clip((bmi_live - 10) / (50 - 10), 0, 1))
        st.progress(bmi_pct)
        bc1, bc2, bc3, bc4 = st.columns(4)
        bc1.markdown("🟡 **<18.5** Underweight")
        bc2.markdown("🟢 **18.5–24.9** Normal")
        bc3.markdown("🔴 **25–29.9** Overweight")
        bc4.markdown("🔴 **≥30** Obese")
        st.markdown("<br>", unsafe_allow_html=True)

        # ── Workout & Diet ────────────────────────────────────────────
        col_w, col_d = st.columns(2)
        with col_w:
            st.markdown('<div class="section-title">🏋️ Workout Plan</div>', unsafe_allow_html=True)
            st.markdown(f"""<div class="rec-box">
              <h4>Recommended for: <em>{ml_category}</em></h4>
              <ul>{"".join(f"<li>{w}</li>" for w in rec["workout"])}</ul>
            </div>""", unsafe_allow_html=True)

        with col_d:
            st.markdown('<div class="section-title">🥗 Diet Plan</div>', unsafe_allow_html=True)
            st.markdown(f"""<div class="rec-box">
              <h4>Nutrition for: <em>{ml_category}</em></h4>
              <ul>{"".join(f"<li>{d}</li>" for d in rec["diet"])}</ul>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Calorie Goal box ──────────────────────────────────────────
        st.markdown('<div class="section-title">🔥 Daily Calorie Target</div>', unsafe_allow_html=True)
        st.info(f"**{rec['calorie_goal']}**  \n{GOAL_TIPS[fitness_goal]}")

        # ── Weekly schedule ───────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">📅 Your 7‑Day Schedule</div>', unsafe_allow_html=True)
        sched = pd.DataFrame({
            "Day":      list(rec["weekly"].keys()),
            "Activity": list(rec["weekly"].values()),
        })
        st.dataframe(sched, hide_index=True)

        # ── Extra data-driven insight from dataset ────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">📈 Dataset Benchmark for Your Profile</div>',
                    unsafe_allow_html=True)
        similar = df[
            (df["Age"].between(age - 5, age + 5)) &
            (df["Gender"] == gender) &
            (df["BMI_Category"] == ml_category)
        ]
        if len(similar) > 0:
            s1, s2, s3 = st.columns(3)
            with s1:
                st.markdown(f"""<div class="metric-card">
                  <div class="metric-value" style="color:#a78bfa;">{len(similar)}</div>
                  <div class="metric-label">Similar profiles in dataset</div>
                </div>""", unsafe_allow_html=True)
            with s2:
                st.markdown(f"""<div class="metric-card">
                  <div class="metric-value" style="color:#34d399;">{similar['Calories Burn'].mean():.0f} kcal</div>
                  <div class="metric-label">Avg Calories Burned (dataset peers)</div>
                </div>""", unsafe_allow_html=True)
            with s3:
                st.markdown(f"""<div class="metric-card">
                  <div class="metric-value" style="color:#60a5fa;">{similar['Duration'].mean():.0f} min</div>
                  <div class="metric-label">Avg Exercise Duration (dataset peers)</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.warning("No closely matching profiles found in the dataset for benchmark comparison.")


# ════════════════════════════════════════════════
# TAB 2 – DATASET INSIGHTS
# ════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">📊 Dataset Overview – exercise_dataset.csv</div>',
                unsafe_allow_html=True)

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-value" style="color:#60a5fa;">{len(df):,}</div>
          <div class="metric-label">Total Records</div>
        </div>""", unsafe_allow_html=True)
    with d2:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-value" style="color:#34d399;">{df['Gender'].value_counts().get('Male',0):,}</div>
          <div class="metric-label">Male Participants</div>
        </div>""", unsafe_allow_html=True)
    with d3:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-value" style="color:#f472b6;">{df['Gender'].value_counts().get('Female',0):,}</div>
          <div class="metric-label">Female Participants</div>
        </div>""", unsafe_allow_html=True)
    with d4:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-value" style="color:#fbbf24;">{df['Calories Burn'].mean():.0f} kcal</div>
          <div class="metric-label">Avg Calories Burned</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row 1
    cl1, cl2 = st.columns(2)
    with cl1:
        st.markdown('<div class="section-title">📈 Age Distribution</div>', unsafe_allow_html=True)
        age_series = pd.cut(df["Age"], bins=[10, 20, 30, 40, 50, 60, 70, 80])
        age_counts_s = age_series.value_counts().sort_index()
        age_df = pd.DataFrame({"Count": age_counts_s.values},
                               index=age_counts_s.index.astype(str))
        st.bar_chart(age_df)

    with cl2:
        st.markdown('<div class="section-title">🔥 Avg Calories by Exercise Intensity</div>', unsafe_allow_html=True)
        intens = df.groupby("Exercise Intensity")["Calories Burn"].mean()
        st.bar_chart(pd.DataFrame({"Avg Calories": intens}))

    # Charts row 2
    cl3, cl4 = st.columns(2)
    with cl3:
        st.markdown('<div class="section-title">⚖️ BMI Category Distribution</div>', unsafe_allow_html=True)
        bmi_dist = df["BMI_Category"].value_counts()
        st.bar_chart(pd.DataFrame({"Count": bmi_dist}))

    with cl4:
        st.markdown('<div class="section-title">⏱️ Avg Duration by Gender</div>', unsafe_allow_html=True)
        dur_g = df.groupby("Gender")["Duration"].mean()
        st.bar_chart(pd.DataFrame({"Avg Duration (min)": dur_g}))

    # Charts row 3
    cl5, cl6 = st.columns(2)
    with cl5:
        st.markdown('<div class="section-title">❤️ Avg Heart Rate by BMI Category</div>', unsafe_allow_html=True)
        hr_bmi = df.groupby("BMI_Category")["Heart Rate"].mean()
        st.bar_chart(pd.DataFrame({"Avg Heart Rate": hr_bmi}))

    with cl6:
        st.markdown('<div class="section-title">🎯 Dream Weight vs Actual Weight</div>', unsafe_allow_html=True)
        wt_df = df[["Actual Weight", "Dream Weight"]].describe().loc[["mean", "min", "max"]].T
        st.dataframe(wt_df.round(2), use_container_width=True)
        st.caption(f"Average weight gap: **{df['Weight_Diff'].mean():.1f} kg** (Actual – Dream)")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🗂️ Raw Dataset (first 20 rows)</div>', unsafe_allow_html=True)
    display_cols = ["ID", "Age", "Gender", "Actual Weight", "Dream Weight",
                    "BMI", "BMI_Category", "Duration", "Heart Rate",
                    "Exercise Intensity", "Calories Burn"]
    st.dataframe(df[display_cols].head(20), hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 Descriptive Statistics</div>', unsafe_allow_html=True)
    num_cols = ["Age", "Actual Weight", "Dream Weight", "BMI",
                "Duration", "Heart Rate", "Exercise Intensity", "Calories Burn"]
    st.dataframe(df[num_cols].describe().round(2))


# ════════════════════════════════════════════════
# TAB 3 – MODEL PERFORMANCE
# ════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">🤖 Model Evaluation Results</div>', unsafe_allow_html=True)

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-value" style="color:#34d399;">{clf_acc*100:.1f}%</div>
          <div class="metric-label">Decision Tree Accuracy<br><small>(Test Set)</small></div>
        </div>""", unsafe_allow_html=True)
    with mc2:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-value" style="color:#60a5fa;">{reg_r2:.4f}</div>
          <div class="metric-label">Linear Regression R² Score</div>
        </div>""", unsafe_allow_html=True)
    with mc3:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-value" style="color:#f87171;">{reg_mae:.1f} kcal</div>
          <div class="metric-label">Calorie Prediction MAE</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_t1, col_t2 = st.columns(2)

    # ── Decision Tree details ──────────────────────────────────────────
    with col_t1:
        st.markdown('<div class="section-title">🌳 Decision Tree Classifier</div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="rec-box">
          <h4>Model Configuration</h4>
          <ul>
            <li><b>Algorithm:</b> Decision Tree (CART)</li>
            <li><b>Max Depth:</b> 6</li>
            <li><b>Min Samples Leaf:</b> 5</li>
            <li><b>Train / Test Split:</b> 80% / 20% (stratified)</li>
            <li><b>Target Classes:</b> {', '.join(clf_classes)}</li>
          </ul>
          <h4 style="margin-top:.8rem;">Input Features</h4>
          <ul>{"".join(f"<li>{f}</li>" for f in clf_features)}</ul>
        </div>""", unsafe_allow_html=True)

        # Classification report table
        st.markdown("**Per-Class Metrics (Test Set)**")
        report_rows = []
        for cls in clf_classes:
            if cls in clf_report:
                r = clf_report[cls]
                report_rows.append({
                    "Class": cls,
                    "Precision": f"{r['precision']:.2f}",
                    "Recall": f"{r['recall']:.2f}",
                    "F1-Score": f"{r['f1-score']:.2f}",
                    "Support": int(r["support"]),
                })
        st.dataframe(pd.DataFrame(report_rows), hide_index=True)

        # Confusion matrix
        st.markdown("**Confusion Matrix**")
        cm_df = pd.DataFrame(clf_cm, index=clf_classes, columns=clf_classes)
        st.dataframe(cm_df)

    # ── Linear Regression details ──────────────────────────────────────
    with col_t2:
        st.markdown('<div class="section-title">📈 Linear Regression – Calories Burn</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""<div class="rec-box">
          <h4>Model Configuration</h4>
          <ul>
            <li><b>Algorithm:</b> Ordinary Least Squares (OLS)</li>
            <li><b>Feature Scaling:</b> StandardScaler (z-score)</li>
            <li><b>Train / Test Split:</b> 80% / 20%</li>
            <li><b>MAE:</b> {reg_mae:.2f} kcal</li>
            <li><b>RMSE:</b> {reg_rmse:.2f} kcal</li>
            <li><b>R² Score:</b> {reg_r2:.4f}</li>
          </ul>
          <h4 style="margin-top:.8rem;">Input Features</h4>
          <ul>{"".join(f"<li>{f}</li>" for f in reg_features)}</ul>
        </div>""", unsafe_allow_html=True)

        # Coefficients table
        st.markdown("**Feature Coefficients (Scaled)**")
        coef_df = pd.DataFrame({
            "Feature": list(reg_coefs.keys()),
            "Coefficient": list(reg_coefs.values()),
        }).sort_values("Coefficient", key=abs, ascending=False)
        st.dataframe(coef_df, hide_index=True)

        st.markdown("**Calorie Burn – Dataset Distribution Benchmarks**")
        cal_stats = df.groupby("BMI_Category")["Calories Burn"].agg(["mean", "min", "max"]).round(1)
        cal_stats.columns = ["Mean (kcal)", "Min (kcal)", "Max (kcal)"]
        st.dataframe(cal_stats)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📌 Feature Importance – Decision Tree</div>', unsafe_allow_html=True)
    fi_series = pd.Series(clf.feature_importances_, index=clf_features).sort_values(ascending=False)
    st.bar_chart(pd.DataFrame({"Importance": fi_series}))


# ════════════════════════════════════════════════
# TAB 4 – HOW IT WORKS
# ════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">🔄 System Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
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
""")

    st.markdown("<br>", unsafe_allow_html=True)
    col_h1, col_h2 = st.columns(2)

    with col_h1:
        st.markdown('<div class="section-title">📚 Dataset Columns Used</div>', unsafe_allow_html=True)
        cols_info = {
            "ID":                 "Row identifier",
            "Age":                "User age (years) – Model input",
            "Gender":             "Male / Female – encoded for ML",
            "Actual Weight":      "Current body weight (kg) – Model input",
            "Dream Weight":       "Goal body weight (kg) – Model input",
            "BMI":                "Body Mass Index – Primary classification feature",
            "Calories Burn":      "Target for Linear Regression",
            "Duration":           "Exercise session length (min) – Model input",
            "Heart Rate":         "Average HR during session – Model input",
            "Exercise Intensity": "Scale 1–10 – Model input",
            "Weather Conditions": "Contextual (not used in models)",
            "Exercise":           "Exercise type label (contextual)",
        }
        rows = [{"Column": k, "Description": v} for k, v in cols_info.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True)

    with col_h2:
        st.markdown('<div class="section-title">🧮 BMI Formula</div>', unsafe_allow_html=True)
        st.markdown("""
<div class="rec-box">
  <h4>BMI Calculation</h4>
  <p style="color:#e2e8f0; font-size:1.3rem; text-align:center; letter-spacing:.05rem;">
    BMI = Weight (kg) ÷ Height² (m²)
  </p>
  <h4 style="margin-top:.8rem;">Classification Thresholds</h4>
  <ul>
    <li><span class="badge badge-under">Underweight</span>  BMI &lt; 18.5</li>
    <li><span class="badge badge-normal">Normal</span>      18.5 ≤ BMI &lt; 25.0</li>
    <li><span class="badge badge-over">Overweight</span>   BMI ≥ 25.0</li>
  </ul>
</div>
""", unsafe_allow_html=True)

        st.markdown('<div class="section-title" style="margin-top:1rem;">🛠️ Tech Stack</div>',
                    unsafe_allow_html=True)
        tech = {
            "Language":  "Python 3.12",
            "UI":        "Streamlit",
            "ML":        "scikit-learn (DecisionTree, LinearRegression, StandardScaler)",
            "Data":      "pandas, numpy",
            "Dataset":   "exercise_dataset.csv (3,864 rows × 12 columns)",
        }
        st.dataframe(pd.DataFrame(list(tech.items()), columns=["Component", "Detail"]),
                     hide_index=True)


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#475569; font-size:.82rem; padding:.5rem 0 1rem;">
  🏋️ <b>FitAI – Fitness Recommendation System</b> &nbsp;|&nbsp;
  Infosys Internship Project &nbsp;|&nbsp;
  ML Models: <b>Decision Tree Classifier + Linear Regression</b> &nbsp;|&nbsp;
  Dataset: <b>Exercise &amp; Fitness Metrics (3,864 records)</b>
</div>
""", unsafe_allow_html=True)
