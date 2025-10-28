import random
import pandas as pd
# HÀM DỰ ĐOÁN MODEL CỦA BẠN

def compute_features_row(age, sex, height, weight, duration, hr, body_temp):
    """Tạo feature row đúng pipeline train"""
    bmi = weight / ((height / 100) ** 2)
    duration_per_heart = duration / hr
    intensity = hr * duration
    temp_per_minute = body_temp / duration
    age_group = pd.cut([age], bins=[0, 30, 45, 60, 100],
                       labels=["18-30", "31-45", "46-60", "61+"])[0]
    return pd.DataFrame([{
        "Age": age, "Sex": sex, "Height": height, "Weight": weight,
        "Duration": duration, "Heart_Rate": hr, "Body_Temp": body_temp,
        "BMI": bmi, "Duration_per_Heart": duration_per_heart,
        "Intensity": intensity, "Temp_per_Minute": temp_per_minute,
        "Age_Group": age_group
    }])

# BINARY SEARCH HELPER
def _binary_search_to_target(model,
                             age, sex, height, weight, body_temp,
                             duration, hr, target_kcal,
                             var="duration", lo=None, hi=None, iters=32):
    """Tìm MIN duration hoặc HR để đạt target_kcal."""

    def predict_kcal(dur, heart):
        row = compute_features_row(age, sex, height, weight, dur, heart, body_temp)
        return model.predict(row)[0]

    if var == "duration":
        lo = max(lo if lo is not None else duration, 1.0)
        hi = hi if hi is not None else 180.0
        if predict_kcal(hi, hr) < target_kcal:
            return None
        l, r = lo, hi
        for _ in range(iters):
            mid = 0.5 * (l + r)
            kcal = predict_kcal(mid, hr)
            if kcal < target_kcal:
                l = mid
            else:
                r = mid
        return round(r, 1), float(predict_kcal(r, hr))

    else:  # var == "hr"
        lo = max(lo if lo is not None else hr, 60.0)
        hi = hi if hi is not None else 190.0
        if predict_kcal(duration, hi) < target_kcal:
            return None
        l, r = lo, hi
        for _ in range(iters):
            mid = 0.5 * (l + r)
            kcal = predict_kcal(duration, mid)
            if kcal < target_kcal:
                l = mid
            else:
                r = mid
        return round(r, 1), float(predict_kcal(duration, r))


def generate_kitty_tip_with_target_optimization(
    model,
    *,
    age: int,
    sex: str,
    height: float,
    weight: float,
    body_temp: float,
    hr: float,
    duration: float,
    kcal: float,
    target_kcal: float | None = None,
    name: str | None = None,
    max_minutes: float = 180.0,
    max_hr: float = 190.0
) -> str:
    
    # Xưng hô & tone
    call = "bạn nhỏ" if age < 20 else ("cậu" if age < 35 else "bạn")
    pronoun = "chị Kitty" if sex.lower().startswith("nữ") else "Kitty"

    # Đánh giá cường độ
    if hr < 110 or duration < 20:
        level = "nhẹ"
    elif hr < 150 or duration < 50:
        level = "vừa"
    else:
        level = "cao"

    # Tip nền theo cường độ (không dùng icon)
    base_tips = {
        "nhẹ": [
            f"Hôm nay {call} tập nhẹ nhàng và thư giãn. Cơ thể đang được nghỉ ngơi tốt đó.",
            f"{pronoun} thấy {call} giữ nhịp đều, rất tốt."
        ],
        "vừa": [
            f"Cường độ hôm nay vừa phải, đúng chuẩn luôn {call} ơi. Giữ phong độ này nhé.",
            f"{call.capitalize()} đang tiến bộ rõ ràng, Kitty rất vui."
        ],
        "cao": [
            f"{call} rất mạnh mẽ hôm nay. Đã tiêu hao nhiều năng lượng, nhớ nghỉ ngơi đủ nhé.",
            f"Hôm nay {call} tập sung và tràn đầy năng lượng."
        ]
    }
    msg = random.choice(base_tips[level])

    # Nếu không có target → phản hồi chung
    if not target_kcal:
        if kcal > 400:
            msg += f" {pronoun} rất ấn tượng, {kcal:.0f} kcal là con số tuyệt vời."
        elif kcal < 150:
            msg += f" Nhẹ nhàng như vậy là hợp lý. Ngày mai thử tăng thêm 10 phút xem sao."
        else:
            msg += f" Buổi tập hôm nay ổn định lắm {call} ơi. Duy trì đều như vậy là rất tốt."
        return msg.replace(call, name) if name else msg

    # Có target → kiểm tra đạt chưa (không icon)
    diff = kcal - target_kcal
    if diff == 0:
        msg += f" {pronoun} tự hào lắm. {call.capitalize()} vừa đạt đúng mục tiêu hôm nay."
        return msg.replace(call, name) if name else msg
    elif diff > 0:
        msg += f" Rất xuất sắc. {call.capitalize()} đã vượt mục tiêu {diff:.0f} kcal."
        return msg.replace(call, name) if name else msg
    elif -30 <= diff < 0:
        msg += f" Gần chạm mục tiêu rồi. Chỉ còn thiếu một chút nữa thôi, cố thêm một chút nhé."
    else:
        msg += f" Hôm nay còn thiếu {abs(diff):.0f} kcal để đạt mục tiêu. {pronoun} tin {call} làm được, cố lên nhé."

    # Chưa đạt → gợi ý tăng thời lượng
    dur_sol = _binary_search_to_target(
        model, age, sex, height, weight, body_temp,
        duration, hr, target_kcal,
        var="duration", lo=duration, hi=max_minutes
    )

    if dur_sol is not None:
        new_dur, _ = dur_sol
        msg += f"\nKitty gợi ý nhé: chỉ cần tăng thời lượng lên khoảng {new_dur:.0f} phút (giữ nhịp tim {hr:.0f} bpm) là đạt mục tiêu."
        msg += f"\n{pronoun} tin {call} sẽ làm được. Cố gắng nhé!"
        return msg.replace(call, name) if name else msg

    # Nếu duration tối đa vẫn thiếu → gợi ý tăng HR (giữ max_minutes như bạn đã chọn)
    hr_sol = _binary_search_to_target(
        model,
        age, sex, height, weight, body_temp,
        max_minutes, hr, target_kcal,
        var="hr", lo=hr, hi=max_hr
    )

    if hr_sol is not None:
        new_hr, _ = hr_sol
        msg += f"\nKitty gợi ý nhé: nếu giữ {max_minutes:.0f} phút, chỉ cần tăng nhịp tim lên khoảng {new_hr:.0f} bpm là phù hợp."
        msg += f"\nNhớ khởi động kỹ và tăng dần để đảm bảo an toàn."
        return msg.replace(call, name) if name else msg

    # Nếu cả hai vẫn không đủ
    msg += f"\nKitty đã thử các phương án nhưng vẫn chưa chạm được {target_kcal:.0f} kcal trong giới hạn an toàn hôm nay."
    msg += f"\nKhông sao đâu. {call} có thể chia buổi tập làm hai hoặc đặt mục tiêu nhỏ hơn trước."
    return msg.replace(call, name) if name else msg