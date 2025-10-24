import joblib 
MODEL_PATH = "artifacts/best_calorie_model.pkl"
model = joblib.load(MODEL_PATH)

from features_for_web.weekly_planner import weekly_plan_generator
from features_for_web.weekly_planner import UserProfile

free = ["Mon","Tue","Thu","Sat"]

plan = weekly_plan_generator(
    model,
    profile=UserProfile(25,"male",175,70,39),
    weekly_target_kcal=7000,
    days=5,
    max_minutes_per_day=100,
    base_hr=130,
    split_mode="pyramid",
    free_days=free,
    peak_day="Tue"
)
print(plan)

from features_for_web.goal_translator import goal_translator

profile = UserProfile(31, "male", 180, 90, 39)
print()
print(goal_translator(model, profile, 0.7, 0, 100, 140, "duration_first", (100, 180), "pyramid", "Mon", free_days= ["Mon","Tue", "Fri","Sat"], peak_day= "Tue")[0])