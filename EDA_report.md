# Predict Calories Expenditure  
## EDA & Feature Engineering Report

---

## 1. Overview  
- **Train set:** 750,000 rows · 9 columns  
- **Test set:** 250,000 rows  
- **Variables:**  
  - **Numeric:** Age, Height, Weight, Duration, Heart_Rate, Body_Temp, Calories  
  - **Categorical:** Sex  
  - **ID column**  
- **After removing outliers:** 735,027 rows  

---

## 2. Target Variable: Calories  
- **Distribution:** Right-skewed  
- Most sessions burn **10–80 kcal**  
- Fewer high-calorie sessions (**150–300 kcal**)  
- No unusual/extreme outliers  

---

## 3. Correlation Analysis (Heatmap – Numerical Features)  
**Strong positive correlations with Calories:**  
- Duration  
- Heart_Rate  
- Body_Temp  

**Other insights:**  
- Height and Weight are highly correlated with each other  
- Age shows almost no correlation with Calories  

---

## 4. Scatterplot Insights (Features vs Calories)

### Age → Calories  
- No clear trend  
- Age is *not* a meaningful predictor  

### Height → Calories  
- Very weak relationship  
- Points widely scattered  

### Weight → Calories  
- Slight upward trend  
- But still weak predictor  

### Duration → Calories  
- **Strong positive relationship**  
- Longer exercise duration → higher calories burned  

### Heart_Rate → Calories  
- **Clear positive trend**  
- Higher intensity → more calories burned  

### Body_Temp → Calories  
- Positive relationship  
- Higher temperature → more calories burned  

---

## 5. Feature Distributions

### **Age**
- Many participants aged **20–30**  
- Fewer participants aged **60–80**  
- Right-skewed  
- No extreme values  

### **Height**
- Approximately normal distribution  
- Centered at **160–180 cm**  
- Symmetrical with slight tails  
- A few values <150 cm or >200 cm but not severe  

### **Weight**
- **Bimodal-like distribution** (≈65 kg & ≈85 kg peaks)  
- Most between **60–100 kg**  
- Slight right skew due to heavier individuals  

### **Duration**
- Almost **uniform** distribution (1–50 minutes)  
- No skew  
- Balanced sampling of workout lengths  

### **Heart_Rate**
- Normal-like distribution around **90–100 bpm**  
- Symmetrical, smooth tails  

### **Body_Temp**
- **Left-skewed**  
- High counts near **40–41°C**  
- Long left tail toward **37–38°C**  
- High temperatures during exercise are common  

---