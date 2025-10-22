import streamlit as st
import joblib
import pandas as pd

# Load model
MODEL_PATH = "artifacts/best_calorie_model.pkl"
model = joblib.load(MODEL_PATH)
print(model)

# Helper functions

def _age_group(age: float) -> str:
    """Group age to match trained categories."""
    if age <= 30: return "18-30"
    if age <= 45: return "31-45"
    if age <= 60: return "46-60"
    return "61+"


def make_feature_row(age, sex, height_cm, weight_kg, duration_min, heart_rate_bpm, body_temp_c=37.0):
    """Create a single-row DataFrame with all features required by the model."""
    BMI = weight_kg / ((height_cm / 100.0) ** 2)
    Duration_per_Heart = duration_min / heart_rate_bpm if heart_rate_bpm > 0 else 0.0
    Intensity = heart_rate_bpm * duration_min
    Temp_per_Minute = body_temp_c / duration_min if duration_min > 0 else 0.0
    Age_Group = _age_group(age)

    row = {
        "Age": age,
        "Height": height_cm,
        "Weight": weight_kg,
        "Duration": duration_min,
        "Heart_Rate": heart_rate_bpm,
        "Body_Temp": body_temp_c,
        "BMI": BMI,
        "Duration_per_Heart": Duration_per_Heart,
        "Intensity": Intensity,
        "Temp_per_Minute": Temp_per_Minute,
        "Age_Group": Age_Group,
        "Sex": sex
    }
    return pd.DataFrame([row])


def predict_kcal(model, age, sex, height_cm, weight_kg, duration_min, heart_rate_bpm, body_temp_c=37.0):
    """Predict burned calories."""
    X = make_feature_row(age, sex, height_cm, weight_kg, duration_min, heart_rate_bpm, body_temp_c)
    kcal = float(model.predict(X)[0])
    return max(0.0, kcal)


def solve_duration_for_target(model, target_kcal, age, sex, height_cm, weight_kg,
                              base_hr, max_minutes=120, body_temp_c=37.0,
                              tol=0.5, max_iter=40):
    """Binary search to find the duration required to reach target kcal."""
    lo, hi = 0.0, max_minutes
    kcal_hi = predict_kcal(model, age, sex, height_cm, weight_kg, hi, base_hr, body_temp_c)
    if kcal_hi + tol < target_kcal:
        return hi, kcal_hi, False
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        kcal_mid = predict_kcal(model, age, sex, height_cm, weight_kg, mid, base_hr, body_temp_c)
        if abs(kcal_mid - target_kcal) <= tol:
            return mid, kcal_mid, True
        if kcal_mid < target_kcal:
            lo = mid
        else:
            hi = mid
    return hi, predict_kcal(model, age, sex, height_cm, weight_kg, hi, base_hr, body_temp_c), True


# Streamlit UI
st.set_page_config(page_title="Calorie Predictor App", layout="centered")

st.title("Calorie Predictor App")
st.caption("Includes What-if Coach (simulation) and Calorie Swap (reverse calculation).")

# Sidebar selection
feature = st.sidebar.radio(
    "Select Feature:",
    ["What-if Coach (Instant Simulation)", "Calorie Swap (Food ↔ Workout)"]
)

# Sidebar input
st.sidebar.header("Personal Information")
age = st.sidebar.slider("Age", 18, 70, 25)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
height = st.sidebar.number_input("Height (cm)", 140, 210, 170)
weight = st.sidebar.number_input("Weight (kg)", 40, 120, 65)
body_temp = st.sidebar.slider("Body Temperature (°C)", 36.0, 41.0, 38.0)


# FEATURE 1: WHAT-IF COACH

if feature.startswith("What-if"):
    st.header("What-if Coach – Real-time Simulation")

    # User input sliders
    duration = st.slider("Workout Duration (minutes)", 5, 120, 30)
    heart_rate = st.slider("Heart Rate (bpm)", 80, 190, 130)

    # Predict calories
    kcal = predict_kcal(model, age, sex, height, weight, duration, heart_rate, body_temp)
    st.metric("Predicted Calories Burned", f"{kcal:.1f} kcal")

    # Comparison table
    st.subheader("Comparison Scenarios")
    compare_data = []
    for d in [duration - 10, duration, duration + 10]:
        if d > 0:
            k = predict_kcal(model, age, sex, height, weight, d, heart_rate, body_temp)
            compare_data.append({"Scenario": f"{d} min", "Calories": round(k, 1)})
    for hr in [heart_rate - 20, heart_rate, heart_rate + 20]:
        if 60 < hr < 200:
            k = predict_kcal(model, age, sex, height, weight, duration, hr, body_temp)
            compare_data.append({"Scenario": f"{hr} bpm", "Calories": round(k, 1)})

    st.dataframe(pd.DataFrame(compare_data))


# Feature 2: Calorie Swap
if feature.startswith("Calorie Swap"):
    st.header("Calorie Swap – Convert Food Calories to Workout Time")
    st.info("Welcome! Please select a feature and input your data to see results.")

    # User inputs
    food_kcal = st.number_input("Food calories (kcal)", 50, 2000, 500)
    base_hr = st.slider("Average Heart Rate (bpm)", 90, 180, 130)
    max_min = st.slider("Max workout duration (minutes)", 10, 180, 90)

    if st.button("Calculate Required Duration"):
        dur, kcal_est, feasible = solve_duration_for_target(
            model, food_kcal, age, sex, height, weight, base_hr, max_min, body_temp
        )

        if feasible:
            st.success(
                f"You need to work out for {dur:.1f} minutes at {base_hr} bpm to burn {food_kcal:.0f} kcal."
            )
        else:
            st.warning(
                f"Not feasible within {max_min} minutes. You would burn approximately {kcal_est:.1f} kcal."
            )

    st.markdown("---")
    st.caption("Tip: Try adjusting heart rate or max duration for more realistic targets.")
