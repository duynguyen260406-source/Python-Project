from __future__ import annotations
"""
Weekly Activity Planner – Single-file FastAPI backend (FULL DATASET)
Run:
    uvicorn weekly_activity_backend_full:app --reload --port 8000
Docs:
    http://127.0.0.1:8000/docs
"""
from typing import List, Optional, Dict
from io import StringIO
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator



class PredictRequest(BaseModel):
    # Tuỳ mô hình của bạn, ở đây minh hoạ: vector đặc trưng dạng list[float]
    features: List[float]

class PredictResponse(BaseModel):
    ok: bool
    prediction: Optional[float] = None
    error: Optional[str] = None


# --- MODEL (joblib) ---
import os
try:
    import joblib  # pip install joblib
except Exception:
    joblib = None  # vẫn chạy API nếu chưa cài joblib

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/best_calorie_model.pkl")
_model = None
_model_error = None

def _try_load_model():
    """Load model nếu có file; không làm crash nếu thiếu."""
    global _model, _model_error
    if joblib is None:
        _model = None
        _model_error = "joblib chưa được cài (pip install joblib)"
        return
    if not os.path.exists(MODEL_PATH):
        _model = None
        _model_error = f"Không tìm thấy file model tại: {MODEL_PATH}"
        return
    try:
        _model = joblib.load(MODEL_PATH)
        _model_error = None
    except Exception as e:
        _model = None
        _model_error = f"Lỗi khi load model: {e}"

# gọi load khi module khởi tạo
_try_load_model()


# ===== Embedded CSV (100% from your original file) =====
DATA_CSV = r""""Activity, Exercise or Sport (1 hour)",130 lb,155 lb,180 lb,205 lb,Calories per kg
"Cycling, mountain bike, bmx",502,598,695,791,1.75072971940299
"Cycling, <10 mph, leisure bicycling",236,281,327,372,0.823235629850746
"Cycling, >20 mph, racing",944,1126,1308,1489,3.29497352835821
"Cycling, 10-11.9 mph, light",354,422,490,558,1.23485344477612
"Cycling, 12-13.9 mph, moderate",472,563,654,745,1.64782526567164
"Cycling, 14-15.9 mph, vigorous",590,704,817,931,2.05944308059702
"Cycling, 16-19 mph, very fast, racing",708,844,981,1117,2.47106089552239
Unicycling,295,352,409,465,1.02972154029851
"Stationary cycling, very light",177,211,245,279,0.61742672238806
"Stationary cycling, light",325,387,449,512,1.13262599402985
"Stationary cycling, moderate",413,493,572,651,1.44133935522388
"Stationary cycling, vigorous",620,739,858,977,2.16234753432836
"Stationary cycling, very vigorous",738,880,1022,1163,2.57464235223881
"Calisthenics, vigorous, pushups, situps…",472,563,654,745,1.64782526567164
"Calisthenics, light",207,246,286,326,0.721008179104478
"Circuit training, minimal rest",472,563,654,745,1.64782526567164
"Weight lifting, body building, vigorous",354,422,490,558,1.23485344477612
"Weight lifting, light workout",177,211,245,279,0.61742672238806
Health club exercise,325,387,449,512,1.13262599402985
Stair machine,531,633,735,838,1.85295717014925
"Rowing machine, light",207,246,286,326,0.721008179104478
"Rowing machine, moderate",413,493,572,651,1.44133935522388
"Rowing machine, vigorous",502,598,695,791,1.75072971940299
"Rowing machine, very vigorous",708,844,981,1117,2.47106089552239
Ski machine,413,493,572,651,1.44133935522388
"Aerobics, low impact",295,352,409,465,1.02972154029851
"Aerobics, high impact",413,493,572,651,1.44133935522388
"Aerobics, step aerobics",502,598,695,791,1.75072971940299
"Aerobics, general",384,457,531,605,1.33843490149254
Jazzercise,354,422,490,558,1.23485344477612
"Stretching, hatha yoga",236,281,327,372,0.823235629850746
Mild stretching,148,176,204,233,0.515199271641791
Instructing aerobic class,354,422,490,558,1.23485344477612
Water aerobics,236,281,327,372,0.823235629850746
"Ballet, twist, jazz, tap",266,317,368,419,0.927494089552239
"Ballroom dancing, slow",177,211,245,279,0.61742672238806
"Ballroom dancing, fast",325,387,449,512,1.13262599402985
"Running, 5 mph (12 minute mile)",472,563,654,745,1.64782526567164
"Running, 5.2 mph (11.5 minute mile)",531,633,735,838,1.85295717014925
"Running, 6 mph (10 min mile)",590,704,817,931,2.05944308059702
"Running, 6.7 mph (9 min mile)",649,774,899,1024,2.2652519880597
"Running, 7 mph (8.5 min mile)",679,809,940,1070,2.36815644179104
"Running, 7.5mph (8 min mile)",738,880,1022,1163,2.57464235223881
"Running, 8 mph (7.5 min mile)",797,950,1103,1256,2.77977425671642
"Running, 8.6 mph (7 min mile)",826,985,1144,1303,2.88267871044776
"Running, 9 mph (6.5 min mile)",885,1056,1226,1396,3.08916462089552
"Running, 10 mph (6 min mile)",944,1126,1308,1489,3.29497352835821
"Running, 10.9 mph (5.5 min mile)",1062,1267,1471,1675,3.70659134328358
"Running, cross country",531,633,735,838,1.85295717014925
"Running, general",472,563,654,745,1.64782526567164
"Running, on a track, team practice",590,704,817,931,2.05944308059702
"Running, stairs, up",885,1056,1226,1396,3.08916462089552
"Track and field (shot, discus)",236,281,327,372,0.823235629850746
"Track and field (high jump, pole vault)",354,422,490,558,1.23485344477612
Track and field (hurdles),590,704,817,931,2.05944308059702
Archery,207,246,286,326,0.721008179104478
Badminton,266,317,368,419,0.927494089552239
"Basketball game, competitive",472,563,654,745,1.64782526567164
"Playing basketball, non game",354,422,490,558,1.23485344477612
"Basketball, officiating",413,493,572,651,1.44133935522388
"Basketball, shooting baskets",266,317,368,419,0.927494089552239
"Basketball, wheelchair",384,457,531,605,1.33843490149254
"Running, training, pushing wheelchair",472,563,654,745,1.64782526567164
Billiards,148,176,204,233,0.515199271641791
Bowling,177,211,245,279,0.61742672238806
"Boxing, in ring",708,844,981,1117,2.47106089552239
"Boxing, punching bag",354,422,490,558,1.23485344477612
"Boxing, sparring",531,633,735,838,1.85295717014925
"Coaching: football, basketball, soccer…",236,281,327,372,0.823235629850746
"Cricket (batting, bowling)",295,352,409,465,1.02972154029851
Croquet,148,176,204,233,0.515199271641791
Curling,236,281,327,372,0.823235629850746
Darts (wall or lawn),148,176,204,233,0.515199271641791
Fencing,354,422,490,558,1.23485344477612
"Football, competitive",531,633,735,838,1.85295717014925
"Football, touch, flag, general",472,563,654,745,1.64782526567164
"Football or baseball, playing catch",148,176,204,233,0.515199271641791
"Frisbee playing, general",177,211,245,279,0.61742672238806
"Frisbee, ultimate frisbee",472,563,654,745,1.64782526567164
"Golf, general",266,317,368,419,0.927494089552239
"Golf, walking and carrying clubs",266,317,368,419,0.927494089552239
"Golf, driving range",177,211,245,279,0.61742672238806
"Golf, miniature golf",177,211,245,279,0.61742672238806
"Golf, walking and pulling clubs",254,303,351,400,0.885519904477612
"Golf, using power cart",207,246,286,326,0.721008179104478
Gymnastics,236,281,327,372,0.823235629850746
Hacky sack,236,281,327,372,0.823235629850746
Handball,708,844,981,1117,2.47106089552239
"Handball, team",472,563,654,745,1.64782526567164
"Hockey, field hockey",472,563,654,745,1.64782526567164
"Hockey, ice hockey",472,563,654,745,1.64782526567164
"Riding a horse, general",236,281,327,372,0.823235629850746
"Horesback riding, saddling horse",207,246,286,326,0.721008179104478
"Horseback riding, grooming horse",207,246,286,326,0.721008179104478
"Horseback riding, trotting",384,457,531,605,1.33843490149254
"Horseback riding, walking",148,176,204,233,0.515199271641791
"Horse racing, galloping",472,563,654,745,1.64782526567164
"Horse grooming, moderate",354,422,490,558,1.23485344477612
Horseshoe pitching,177,211,245,279,0.61742672238806
Jai alai,708,844,981,1117,2.47106089552239
"Martial arts, judo, karate, jujitsu",590,704,817,931,2.05944308059702
"Martial arts, kick boxing",590,704,817,931,2.05944308059702
"Martial arts, tae kwan do",590,704,817,931,2.05944308059702
Krav maga training,590,704,817,931,2.05944308059702
Juggling,236,281,327,372,0.823235629850746
Kickball,413,493,572,651,1.44133935522388
Lacrosse,472,563,654,745,1.64782526567164
Orienteering,531,633,735,838,1.85295717014925
Playing paddleball,354,422,490,558,1.23485344477612
"Paddleball, competitive",590,704,817,931,2.05944308059702
Polo,472,563,654,745,1.64782526567164
"Racquetball, competitive",590,704,817,931,2.05944308059702
Playing racquetball,413,493,572,651,1.44133935522388
"Rock climbing, ascending rock",649,774,899,1024,2.2652519880597
"Rock climbing, rappelling",472,563,654,745,1.64782526567164
"Jumping rope, fast",708,844,981,1117,2.47106089552239
"Jumping rope, moderate",590,704,817,931,2.05944308059702
"Jumping rope, slow",472,563,654,745,1.64782526567164
Rugby,590,704,817,931,2.05944308059702
"Shuffleboard, lawn bowling",177,211,245,279,0.61742672238806
Skateboarding,295,352,409,465,1.02972154029851
Roller skating,413,493,572,651,1.44133935522388
"Roller blading, in-line skating",708,844,981,1117,2.47106089552239
Sky diving,177,211,245,279,0.61742672238806
"Soccer, competitive",590,704,817,931,2.05944308059702
Playing soccer,413,493,572,651,1.44133935522388
Softball or baseball,295,352,409,465,1.02972154029851
"Softball, officiating",236,281,327,372,0.823235629850746
"Softball, pitching",354,422,490,558,1.23485344477612
Squash,708,844,981,1117,2.47106089552239
"Table tennis, ping pong",236,281,327,372,0.823235629850746
Tai chi,236,281,327,372,0.823235629850746
Playing tennis,413,493,572,651,1.44133935522388
"Tennis, doubles",354,422,490,558,1.23485344477612
"Tennis, singles",472,563,654,745,1.64782526567164
Trampoline,207,246,286,326,0.721008179104478
"Volleyball, competitive",472,563,654,745,1.64782526567164
Playing volleyball,177,211,245,279,0.61742672238806
"Volleyball, beach",472,563,654,745,1.64782526567164
Wrestling,354,422,490,558,1.23485344477612
Wallyball,413,493,572,651,1.44133935522388
"Backpacking, Hiking with pack",413,493,572,651,1.44133935522388
"Carrying infant, level ground",207,246,286,326,0.721008179104478
"Carrying infant, upstairs",295,352,409,465,1.02972154029851
"Carrying 16 to 24 lbs, upstairs",354,422,490,558,1.23485344477612
"Carrying 25 to 49 lbs, upstairs",472,563,654,745,1.64782526567164
"Standing, playing with children, light",165,197,229,261,0.576806543283582
"Walk/run, playing with children, moderate",236,281,327,372,0.823235629850746
"Walk/run, playing with children, vigorous",295,352,409,465,1.02972154029851
Carrying small children,177,211,245,279,0.61742672238806
"Loading, unloading car",177,211,245,279,0.61742672238806
"Climbing hills, carrying up to 9 lbs",413,493,572,651,1.44133935522388
"Climbing hills, carrying 10 to 20 lb",443,528,613,698,1.5449208119403
"Climbing hills, carrying 21 to 42 lb",472,563,654,745,1.64782526567164
"Climbing hills, carrying over 42 lb",531,633,735,838,1.85295717014925
Walking downstairs,177,211,245,279,0.61742672238806
"Hiking, cross country",354,422,490,558,1.23485344477612
Bird watching,148,176,204,233,0.515199271641791
"Marching, rapidly, military",384,457,531,605,1.33843490149254
"Children's games, hopscotch, dodgeball",295,352,409,465,1.02972154029851
Pushing stroller or walking with children,148,176,204,233,0.515199271641791
Pushing a wheelchair,236,281,327,372,0.823235629850746
Race walking,384,457,531,605,1.33843490149254
"Rock climbing, mountain climbing",472,563,654,745,1.64782526567164
Walking using crutches,295,352,409,465,1.02972154029851
Walking the dog,177,211,245,279,0.61742672238806
"Walking, under 2.0 mph, very slow",118,141,163,186,0.411617814925373
"Walking 2.0 mph, slow",148,176,204,233,0.515199271641791
Walking 2.5 mph,177,211,245,279,0.61742672238806
"Walking 3.0 mph, moderate",195,232,270,307,0.679710997014925
"Walking 3.5 mph, brisk pace",224,267,311,354,0.782615450746269
"Walking 3.5 mph, uphill",354,422,490,558,1.23485344477612
"Walking 4.0 mph, very brisk",295,352,409,465,1.02972154029851
Walking 4.5 mph,372,443,515,586,1.29713771940299
Walking 5.0 mph,472,563,654,745,1.64782526567164
"Boating, power, speed boat",148,176,204,233,0.515199271641791
"Canoeing, camping trip",236,281,327,372,0.823235629850746
"Canoeing, rowing, light",177,211,245,279,0.61742672238806
"Canoeing, rowing, moderate",413,493,572,651,1.44133935522388
"Canoeing, rowing, vigorous",708,844,981,1117,2.47106089552239
"Crew, sculling, rowing, competition",708,844,981,1117,2.47106089552239
Kayaking,295,352,409,465,1.02972154029851
Paddle boat,236,281,327,372,0.823235629850746
"Windsurfing, sailing",177,211,245,279,0.61742672238806
"Sailing, competition",295,352,409,465,1.02972154029851
"Sailing, yachting, ocean sailing",177,211,245,279,0.61742672238806
"Skiing, water skiing",354,422,490,558,1.23485344477612
Ski mobiling,413,493,572,651,1.44133935522388
"Skin diving, fast",944,1126,1308,1489,3.29497352835821
"Skin diving, moderate",738,880,1022,1163,2.57464235223881
"Skin diving, scuba diving",413,493,572,651,1.44133935522388
Snorkeling,295,352,409,465,1.02972154029851
"Surfing, body surfing or board surfing",177,211,245,279,0.61742672238806
"Whitewater rafting, kayaking, canoeing",295,352,409,465,1.02972154029851
"Swimming laps, freestyle, fast",590,704,817,931,2.05944308059702
"Swimming laps, freestyle, slow",413,493,572,651,1.44133935522388
Swimming backstroke,413,493,572,651,1.44133935522388
Swimming breaststroke,590,704,817,931,2.05944308059702
Swimming butterfly,649,774,899,1024,2.2652519880597
"Swimming leisurely, not laps",354,422,490,558,1.23485344477612
Swimming sidestroke,472,563,654,745,1.64782526567164
Swimming synchronized,472,563,654,745,1.64782526567164
"Swimming, treading water, fast, vigorous",590,704,817,931,2.05944308059702
"Swimming, treading water, moderate",236,281,327,372,0.823235629850746
"Water aerobics, water calisthenics",236,281,327,372,0.823235629850746
Water polo,590,704,817,931,2.05944308059702
Water volleyball,177,211,245,279,0.61742672238806
Water jogging,472,563,654,745,1.64782526567164
"Diving, springboard or platform",177,211,245,279,0.61742672238806
"Ice skating, < 9 mph",325,387,449,512,1.13262599402985
"Ice skating, average speed",413,493,572,651,1.44133935522388
"Ice skating, rapidly",531,633,735,838,1.85295717014925
"Speed skating, ice, competitive",885,1056,1226,1396,3.08916462089552
"Cross country snow skiing, slow",413,493,572,651,1.44133935522388
"Cross country skiing, moderate",472,563,654,745,1.64782526567164
"Cross country skiing, vigorous",531,633,735,838,1.85295717014925
"Cross country skiing, racing",826,985,1144,1303,2.88267871044776
"Cross country skiing, uphill",974,1161,1348,1536,3.39787798208955
"Snow skiing, downhill skiing, light",295,352,409,465,1.02972154029851
"Downhill snow skiing, moderate",354,422,490,558,1.23485344477612
"Downhill snow skiing, racing",472,563,654,745,1.64782526567164
"Sledding, tobagganing, luge",413,493,572,651,1.44133935522388
Snow shoeing,472,563,654,745,1.64782526567164
Snowmobiling,207,246,286,326,0.721008179104478
General housework,207,246,286,326,0.721008179104478
Cleaning gutters,295,352,409,465,1.02972154029851
Painting,266,317,368,419,0.927494089552239
"Sit, playing with animals",148,176,204,233,0.515199271641791
"Walk / run, playing with animals",236,281,327,372,0.823235629850746
Bathing dog,207,246,286,326,0.721008179104478
"Mowing lawn, walk, power mower",325,387,449,512,1.13262599402985
"Mowing lawn, riding mower",148,176,204,233,0.515199271641791
"Walking, snow blower",207,246,286,326,0.721008179104478
"Riding, snow blower",177,211,245,279,0.61742672238806
Shoveling snow by hand,354,422,490,558,1.23485344477612
Raking lawn,254,303,351,400,0.885519904477612
"Gardening, general",236,281,327,372,0.823235629850746
"Bagging grass, leaves",236,281,327,372,0.823235629850746
Watering lawn or garden,89,106,123,140,0.310067367164179
"Weeding, cultivating garden",266,317,368,419,0.927494089552239
"Carpentry, general",207,246,286,326,0.721008179104478
Carrying heavy loads,472,563,654,745,1.64782526567164
Carrying moderate loads upstairs,472,563,654,745,1.64782526567164
General cleaning,207,246,286,326,0.721008179104478
"Cleaning, dusting",148,176,204,233,0.515199271641791
Taking out trash,177,211,245,279,0.61742672238806
"Walking, pushing a wheelchair",236,281,327,372,0.823235629850746
"Teach physical education,exercise class",236,281,327,372,0.823235629850746
"""

VN_DAYS = ["Th\u1ee9 hai", "Th\u1ee9 ba", "Th\u1ee9 t\u01b0", "Th\u1ee9 n\u0103m", "Th\u1ee9 s\u00e1u", "Th\u1ee9 b\u1ea3y", "Ch\u1ee7 nh\u1eadt"]
VN_GROUPS = ["trong nhà","ngoài trời","thể thao","nghệ thuật","kháng lực"]
DAY_TO_INDEX: Dict[str, int] = {d.lower(): i for i, d in enumerate(VN_DAYS)}

def load_embedded_dataframe() -> pd.DataFrame:
    return pd.read_csv(StringIO(DATA_CSV))

def classify_group(activity_name: str) -> str:
    name = str(activity_name).lower()
    keywords = {
        "trong nhà": [
            "stationary","indoor","home","house","domestic","treadmill",
            "elliptical","stair","cooking","cleaning","laundry","vacuum"
        ],
        "ngoài trời": [
            "cycling","bicycl","hiking","ski","skate","kayak","canoe",
            "climb","surf","row","paddle","gardening","mowing","yard","hunting"
        ],
        "thể thao": [
            "running","jogging","basketball","football","soccer","tennis",
            "badminton","volleyball","swim","boxing","martial","rugby",
            "baseball","softball","handball","racquetball","squash"
        ],
        "nghệ thuật": ["dance","ballet","aerobic dance","zumba"],
        "kháng lực": ["weight","strength","resistance","bodybuilding","calisthenics","pilates"],
    }
    for group, kws in keywords.items():
        if any(kw in name for kw in kws):
            return group
    if "cycle" in name or "bike" in name:
        return "ngoài trời"
    if "run" in name:
        return "thể thao"
    if "dance" in name:
        return "nghệ thuật"
    if "weight" in name or "strength" in name:
        return "kháng lực"
    return "ngoài trời"

def add_group_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Group"] = df["Activity, Exercise or Sport (1 hour)"].apply(classify_group)
    return df

def pick_k_activities(df_grouped: pd.DataFrame, k: int = 2, seed: int = 42):
    rng = np.random.default_rng(seed)
    g = df_grouped.sort_values("Calories per kg").reset_index(drop=True)
    if len(g) == 0:
        return []
    if len(g) <= k:
        return [g.iloc[i] for i in range(len(g))]
    picks = []
    mid = len(g) // 2
    idx_high = int(rng.integers(mid, len(g)))
    idx_low  = int(rng.integers(0, max(1, mid)))
    picks.append(g.iloc[idx_high])
    if len(g) > 1:
        if g.iloc[idx_low]["Activity, Exercise or Sport (1 hour)"] == g.iloc[idx_high]["Activity, Exercise or Sport (1 hour)"]:
            idx_low = (idx_low + 1) % len(g)
        picks.append(g.iloc[idx_low])
    remain_idx = [i for i in range(len(g)) if i not in [idx_high, idx_low]]
    rng.shuffle(remain_idx)
    for i in remain_idx:
        if len(picks) >= k:
            break
        picks.append(g.iloc[i])
    return picks

def greedy_adjust_minutes(minutes, cpms, target_kcal, active_mask, min_m=60.0, max_m=120.0, max_iter=800):
    minutes = minutes.astype(float).copy()
    cpms = cpms.astype(float).copy()
    eps = 0.5
    for _ in range(max_iter):
        total = float(np.sum(minutes * cpms * active_mask))
        gap = target_kcal - total
        if abs(gap) <= eps:
            break
        if gap > 0:
            order = np.argsort(-cpms)
            progressed = False
            for i in order:
                if not active_mask[i]:
                    continue
                if minutes[i] < max_m:
                    room = max_m - minutes[i]
                    need = gap / cpms[i] if cpms[i] > 0 else room
                    delta = min(room, max(0.0, need))
                    if delta > 0:
                        minutes[i] += delta
                        progressed = True
                        break
            if not progressed:
                break
        else:
            order = np.argsort(-cpms)
            progressed = False
            for i in order:
                if not active_mask[i]:
                    continue
                if minutes[i] > min_m:
                    room = minutes[i] - min_m
                    need = (-gap) / cpms[i] if cpms[i] > 0 else room
                    delta = min(room, max(0.0, need))
                    if delta > 0:
                        minutes[i] -= delta
                        progressed = True
                        break
            if not progressed:
                break
    return minutes

def ensure_heavy_day_max(minutes, cpms, active_mask, heavy_idx, min_m=60.0, max_m=120.0):
    minutes = minutes.copy()
    kcal = minutes * cpms * active_mask
    others = np.where((np.arange(7) != heavy_idx) & (active_mask > 0), kcal, -np.inf)
    j = int(np.argmax(others))
    if kcal[heavy_idx] < others[j]:
        if minutes[heavy_idx] < max_m:
            minutes[heavy_idx] = min(max_m, minutes[heavy_idx] + (max_m - minutes[heavy_idx]) * 0.8)
        kcal = minutes * cpms * active_mask
        if kcal[heavy_idx] < kcal[j]:
            diff = kcal[j] - kcal[heavy_idx]
            reducible = max(0, minutes[j] - min_m)
            addable = max(0, max_m - minutes[heavy_idx])
            delta = min(reducible, addable, diff / max(cpms[j], 1e-6))
            minutes[j] = max(min_m, minutes[j] - delta)
            minutes[heavy_idx] = min(max_m, minutes[heavy_idx] + delta)
    return minutes

def generate_plan(
    weight_kg: float,
    weekly_target: float,
    groups_selected: List[str],
    days_active: List[str],
    heavy_day: str,
    seed: int = 42,
    min_minutes: float = 60.0,
    max_minutes: float = 120.0,
) -> pd.DataFrame:
    df = add_group_column(load_embedded_dataframe())
    df_g1 = df[df["Group"] == groups_selected[0]].copy()
    df_g2 = df[df["Group"] == groups_selected[1]].copy()
    if len(df_g1) == 0 or len(df_g2) == 0:
        raise ValueError("Một trong hai nhóm không có hoạt động nào. Hãy chọn nhóm khác.")
    acts_g1 = pick_k_activities(df_g1, k=2, seed=seed)
    acts_g2 = pick_k_activities(df_g2, k=2, seed=seed+1)
    pool = acts_g1 + acts_g2
    names = [str(a["Activity, Exercise or Sport (1 hour)"]) for a in pool]
    calkg = [float(a["Calories per kg"]) for a in pool]
    cpms_map = {n: (ck * float(weight_kg)) / 60.0 for n, ck in zip(names, calkg)}
    active_mask = np.array([(d in days_active) for d in VN_DAYS], dtype=bool)
    assignment = ["rest"] * 7
    pool_idx = list(range(len(names)))
    ai = 0
    active_positions = [i for i in range(7) if active_mask[i]]
    for pos in active_positions:
        tried = set()
        chosen = None
        for _ in range(len(pool_idx) * 2):
            cand = pool_idx[ai % len(pool_idx)]
            ai += 1
            if cand in tried:
                continue
            tried.add(cand)
            assignment[pos] = names[cand]
            ok = True
            for j in range(max(0, pos-2), min(6, pos)+1):
                window = list(range(j, min(j+3, 7)))
                acts = [assignment[w] for w in window if assignment[w] != "rest"]
                if len(acts) >= 3 and len(set(acts)) < 3:
                    ok = False
                    break
            if ok:
                chosen = cand
                break
        if chosen is None:
            assignment[pos] = names[pool_idx[(ai-1) % len(pool_idx)]]

    heavy_idx = DAY_TO_INDEX[heavy_day.lower()]
    max_cal_activity = max(zip(names, calkg), key=lambda x: x[1])[0]
    assignment[heavy_idx] = max_cal_activity

    minutes = np.zeros(7, dtype=float)
    for i in range(7):
        minutes[i] = 0.0 if assignment[i] == "rest" else (110.0 if i == heavy_idx else 90.0)
    minutes = np.clip(minutes, 0.0, max_minutes)

    cpms = np.array([cpms_map.get(assignment[i], 0.0) for i in range(7)], dtype=float)

    base_total = float(np.sum(minutes * cpms * (active_mask.astype(float))))
    if base_total > 0:
        scale = weekly_target / base_total
        minutes = minutes * scale

    for i in range(7):
        if assignment[i] == "rest":
            minutes[i] = 0.0
        else:
            minutes[i] = float(np.clip(minutes[i], min_minutes, max_minutes))

    minutes = greedy_adjust_minutes(minutes, cpms, weekly_target, (minutes > 0).astype(float), min_minutes, max_minutes)
    minutes = ensure_heavy_day_max(minutes, cpms, (minutes > 0).astype(float), heavy_idx, min_minutes, max_minutes)

    kcal_per_day = minutes * cpms
    out_df = pd.DataFrame({
        "Ngày": VN_DAYS,
        "Hoạt động": assignment,
        "Thời lượng (phút)": np.round(minutes, 1),
        "Kcal/ngày": np.round(kcal_per_day, 1),
    })
    return out_df

class PlanRequest(BaseModel):
    weight_kg: float = Field(ge=35.0, le=200.0, description="Cân nặng (kg)")
    weekly_target: float = Field(ge=500.0, le=100000.0, description="Mục tiêu kcal/tuần")
    groups_selected: List[str]
    days_active: List[str]
    heavy_day: str
    seed: int = 42
    min_minutes: float = 60.0
    max_minutes: float = 120.0
    @field_validator("groups_selected")
    @classmethod
    def _chk_groups(cls, v: List[str]):
        if len(v) != 2:
            raise ValueError("Bạn phải chọn đúng 2 nhóm hoạt động")
        wrong = [g for g in v if g not in ["trong nhà","ngoài trời","thể thao","nghệ thuật","kháng lực"]]
        if wrong:
            raise ValueError(f"Nhóm không hợp lệ: {wrong}")
        return v
    @field_validator("days_active")
    @classmethod
    def _chk_days(cls, v: List[str]):
        wrong = [d for d in v if d not in VN_DAYS]
        if wrong:
            raise ValueError(f"Tên ngày không hợp lệ: {wrong}")
        return v
    @field_validator("heavy_day")
    @classmethod
    def _chk_heavy(cls, v: str):
        if v not in VN_DAYS:
            raise ValueError("heavy_day phải là một trong các ngày VN_DAYS")
        return v

class DayPlan(BaseModel):
    ngay: str
    hoat_dong: str
    thoi_luong_phut: float
    kcal_ngay: float

class PlanResponse(BaseModel):
    plan: List[DayPlan]
    total_kcal: float
    weekly_target: float
    selected_groups: List[str]
    heavy_day: str

app = FastAPI(title="Weekly Activity Planner API (Full, single file)", version="1.0.0")

@app.get("/model/status")
def model_status():
    """Kiểm tra tình trạng model hiện tại."""
    return {
        "loaded": _model is not None,
        "model_path": MODEL_PATH,
        "error": _model_error,
        "repr": str(_model) if _model is not None else None,
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Gọi dự đoán từ model joblib (nếu có)."""
    # Nếu chưa load được, thử lại (phòng khi bạn vừa thêm file .pkl)
    if _model is None:
        _try_load_model()
        if _model is None:
            return PredictResponse(ok=False, error=_model_error or "Model chưa sẵn sàng")

    try:
        import numpy as np
        X = np.array(req.features, dtype=float).reshape(1, -1)
        y = _model.predict(X)
        val = float(y[0]) if hasattr(y, "__len__") else float(y)
        return PredictResponse(ok=True, prediction=val)
    except Exception as e:
        return PredictResponse(ok=False, error=f"Lỗi dự đoán: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status":"ok"}

@app.get("/activities")
def list_activities():
    df = add_group_column(load_embedded_dataframe())
    return {
        "days": VN_DAYS,
        "groups": ["trong nhà","ngoài trời","thể thao","nghệ thuật","kháng lực"],
        "count": int(len(df)),
        "sample": df["Activity, Exercise or Sport (1 hour)"].head(20).tolist()
    }


@app.post("/plan", response_model=PlanResponse)
def make_plan(req: PlanRequest):
    days_active = list(dict.fromkeys(req.days_active))
    if req.heavy_day not in days_active:
        days_active.append(req.heavy_day)
    try:
        df = generate_plan(
            weight_kg=req.weight_kg,
            weekly_target=req.weekly_target,
            groups_selected=req.groups_selected,
            days_active=days_active,
            heavy_day=req.heavy_day,
            seed=req.seed,
            min_minutes=req.min_minutes,
            max_minutes=req.max_minutes,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    total_kcal = float(df["Kcal/ngày"].sum())
    plan = [
        {
            "ngay": str(row["Ngày"]),
            "hoat_dong": str(row["Hoạt động"]),
            "thoi_luong_phut": float(row["Thời lượng (phút)"]),
            "kcal_ngay": float(row["Kcal/ngày"]),
        }
        for _, row in df.iterrows()
    ]
    return PlanResponse(
        plan=plan,
        total_kcal=total_kcal,
        weekly_target=req.weekly_target,
        selected_groups=req.groups_selected,
        heavy_day=req.heavy_day,
    )
