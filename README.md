# IRIS Classifier with Fairness Analysis

A machine learning project demonstrating **bias detection** and **explainability** using the classic IRIS dataset with an added sensitive attribute (location). This project uses **Fairlearn** for fairness metrics and **SHAP** for model interpretability.

---

## 📁 Project Structure

```
22F3002972_Assignment_9_SEPT_2025_MLOps/
├── src/
│   ├── add_location_attribute.py
│   └── train_with_fairness.py
├── reports/
│   ├── fairness_analysis.png
│   ├── shap_summary_virginica.png
│   └── shap_importance_virginica.png
└── week_9_report.ipynb
```

---

## 📄 File Descriptions

### **`src/` Directory - Source Code**

#### **1. `add_location_attribute.py`** (2 KB)

**Purpose:** Preprocesses the IRIS dataset by adding a simulated geographic location attribute.

**What it does:**
- Loads the original IRIS dataset (`data/iris.csv`)
- Adds a binary `location` attribute (0 or 1) randomly assigned to each sample
- This simulates a **sensitive attribute** that should NOT influence predictions
- Saves the enhanced dataset to `data/iris_with_location.csv`

**Key Functions:**
- `add_location_attribute()`: Main function that adds location and saves data
- Uses `random_state=42` for reproducibility

**Output:**
```
Location distribution:
location
1    82  (54.7%)
0    68  (45.3%)
```

**Why it's important:** Creates a realistic scenario where a sensitive attribute (like geographic location, demographic data) exists in the dataset but should be ignored by a fair model.

**Usage:**
```bash
python src/add_location_attribute.py
```

---

#### **2. `train_with_fairness.py`** (16 KB)

**Purpose:** Main training pipeline with comprehensive fairness analysis and explainability.

**What it does:**
1. **Trains** a Random Forest classifier on IRIS data
2. **Evaluates** overall model performance (accuracy, precision, recall, F1)
3. **Analyzes fairness** across locations using Fairlearn
4. **Explains predictions** using SHAP values for interpretability
5. **Generates visualizations** for stakeholder communication

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `load_and_prepare_data()` | Loads data, splits into train/test sets |
| `train_model()` | Trains Random Forest with specified hyperparameters |
| `evaluate_overall_performance()` | Calculates accuracy, precision, recall, F1 |
| `fairness_analysis()` | Uses Fairlearn to detect bias between locations |
| `shap_explainability()` | Generates SHAP plots for feature importance |
| `explain_shap_plots_simple()` | Provides educational explanation of SHAP plots |

**Fairness Metrics Calculated:**
- **Accuracy by group** (Location 0 vs Location 1)
- **Precision by group**
- **Recall by group**
- **Selection rate by group**
- **Absolute differences** between groups
- **Ratios** between groups

**Fairness Thresholds:**
- ✅ Accuracy difference < 5% → FAIR
- ⚠️ Accuracy difference ≥ 5% → POTENTIAL BIAS
- ✅ Selection rate difference < 10% → FAIR

**SHAP Analysis:**
- Focuses on `virginica` class (configurable)
- Generates **beeswarm plot** showing individual prediction impacts
- Generates **bar plot** showing average feature importance
- Ranks features by importance to detect if `location` is being misused

**Usage:**
```bash
python src/train_with_fairness.py
```

**Expected Output:**
```
Overall Accuracy: 91.11%
Accuracy difference (Location 0 vs 1): 1.19% - FAIR ✓
Location feature importance: 0.0026 (ranked 5/5)
```

---

### **`reports/` Directory - Visualizations**

#### **3. `fairness_analysis.png`** (58 KB)

**Purpose:** Visual comparison of model performance across different locations.

**Contains:**
- **Left plot:** Bar chart showing accuracy by location
  - Location 0: 90.48%
  - Location 1: 91.67%
  - Difference: 1.19% (well below 5% threshold ✅)
  
- **Right plot:** Grouped bar chart comparing all metrics
  - Accuracy, Precision, Recall, Selection Rate
  - Side-by-side comparison of Location 0 vs Location 1

**How to interpret:**
- Similar bar heights = Fair treatment across groups ✅
- Large differences = Potential bias ⚠️

**Use case:** Present to stakeholders to demonstrate model treats both locations equally.

---

#### **4. `shap_summary_virginica.png`** (63 KB)

**Purpose:** Detailed explanation of what influences predictions for the virginica class.

**Visualization type:** Beeswarm plot (dot plot)

**How to read:**
- **Each dot** = One flower sample in the test set
- **Y-axis** = Features (ranked by importance, top = most important)
- **X-axis** = SHAP value (impact on prediction)
  - Right (positive) = Pushes toward "virginica"
  - Left (negative) = Pushes away from "virginica"
- **Color** = Feature value
  - Red = High value (e.g., petal_length = 6.5 cm)
  - Blue = Low value (e.g., petal_length = 1.5 cm)

**Key insights from your plot:**
1. **petal_width**: Red dots on right → High petal width strongly predicts virginica ✅
2. **petal_length**: Similar pattern, high values predict virginica ✅
3. **sepal_length/width**: Mixed patterns, minor contributors
4. **location**: Clustered around zero → No systematic bias! ✅✅✅

**What fairness looks like:**
- Location dots scattered around zero (no clear left/right pattern)
- Mixed red/blue colors (no correlation with feature value)
- Bottom ranking (least important feature)

---

#### **5. `shap_importance_virginica.png`** (50 KB)

**Purpose:** Simple bar chart showing average feature importance.

**How to read:**
- **Longer bar** = More important for predicting virginica
- **Bar length** = Mean absolute SHAP value across all predictions

**Your results:**
1. **petal_width**: ~0.21 (Most important) ✅
2. **petal_length**: ~0.18 (Very important) ✅
3. **sepal_length**: ~0.05 (Minor role)
4. **sepal_width**: ~0.01 (Very minor)
5. **location**: ~0.001 (Nearly zero - FAIR!) ✅✅✅

**Interpretation:**
- Model correctly uses **biological features** (petal measurements)
- Model ignores **location bias** (importance ≈ 0)
- This validates the model is making decisions based on relevant features

---

### **6. `week_9_report.ipynb`** (238 KB)

**Purpose:** Comprehensive Jupyter notebook documenting the entire project.

**Expected contents:**
- **Introduction**: Problem statement, objectives
- **Methodology**: 
  - Data preprocessing steps
  - Model selection rationale
  - Fairness metrics explained
- **Code cells**: Can run the entire pipeline interactively
- **Results**: 
  - Model performance metrics
  - Fairness analysis results
  - SHAP explanations
- **Visualizations**: Embedded plots from reports/
- **Discussion**: 
  - What makes the model fair?
  - Limitations and future work
- **Conclusions**: Key findings and recommendations

**Use cases:**
- Interactive exploration of results
- Educational resource for understanding fairness in ML
- Portfolio piece demonstrating ML ops best practices
- Submission for assignment/project evaluation

---

## 🚀 Quick Start

### **1. Install Dependencies**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn fairlearn shap
```

### **2. Add Location Attribute**
```bash
python src/add_location_attribute.py
```
Creates `data/iris_with_location.csv`

### **3. Run Fairness Analysis**
```bash
python src/train_with_fairness.py
```
Generates plots in `reports/`

### **4. Review Results**
Open `week_9_report.ipynb` in Jupyter to see the full analysis

---

## 📊 Key Results

### **Model Performance**
- **Accuracy**: 91.11%
- **Precision**: 91.55%
- **Recall**: 91.11%
- **F1 Score**: 91.07%

### **Fairness Metrics**
- **Accuracy difference**: 1.19% (FAIR ✅)
- **Location importance**: 0.0026 (effectively zero)
- **Both locations treated equally**

### **Explainability**
- **Top features**: petal_width (0.21), petal_length (0.18)
- **Least important**: location (0.001)
- **Model uses biological features, not sensitive attributes** ✅

---

## 🎯 Why This Project Matters

### **Real-World Applications**
This project demonstrates skills applicable to:

1. **Healthcare ML**: Ensuring medical diagnosis models don't discriminate by race, gender, or location
2. **Financial ML**: Making loan approval models fair across demographics
3. **Hiring ML**: Preventing resume screening tools from biased decisions
4. **Criminal Justice**: Ensuring risk assessment models are equitable

### **Key Takeaways**
✅ Fairness analysis should be standard practice in ML  
✅ Explainability (SHAP) helps detect hidden biases  
✅ Multiple metrics (Fairlearn + SHAP) provide comprehensive view  
✅ Fair models maintain high accuracy while treating groups equally

---

## 📚 Technologies Used

| Technology | Purpose |
|------------|---------|
| **scikit-learn** | Model training (Random Forest) |
| **Fairlearn** | Group fairness metrics |
| **SHAP** | Model explainability and feature importance |
| **pandas** | Data manipulation |
| **matplotlib/seaborn** | Visualizations |
| **numpy** | Numerical operations |

---

## 🔍 Fairness Analysis Workflow

```
┌─────────────────────────┐
│  Original IRIS Dataset  │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Add Location Attribute │ ← add_location_attribute.py
│  (Sensitive Feature)    │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   Train Random Forest   │
│   (5 features)          │
└───────────┬─────────────┘
            │
            ├──────────────────────┐
            │                      │
            ▼                      ▼
    ┌──────────────┐      ┌──────────────┐
    │  Fairlearn   │      │     SHAP     │
    │  Analysis    │      │  Analysis    │
    └──────┬───────┘      └──────┬───────┘
           │                     │
           ▼                     ▼
    ┌──────────────┐      ┌──────────────┐
    │  Group Bias  │      │  Individual  │
    │  Detection   │      │  Feature     │
    │              │      │  Impact      │
    └──────┬───────┘      └──────┬───────┘
           │                     │
           └──────────┬──────────┘
                      │
                      ▼
              ┌───────────────┐
              │  Fairness     │
              │  Verdict:     │
              │  ✅ FAIR!     │
              └───────────────┘
```

---

## 📖 Further Reading

- [Fairlearn Documentation](https://fairlearn.org/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Google's ML Fairness Guide](https://developers.google.com/machine-learning/fairness-overview)
- [IBM AI Fairness 360](https://aif360.mybluemix.net/)

---

## 👤 Author

**Student ID**: 22F3002972  
**Assignment**: Week 9 - ML Fairness & Explainability  
**Course**: MLOps (September 2025)  
**Date**: November 30, 2025

---

## 📝 License

This project is for educational purposes as part of an MLOps course assignment.

---

## 🙏 Acknowledgments

- IRIS dataset: R.A. Fisher (1936)
- Fairlearn library: Microsoft Research
- SHAP library: Scott Lundberg & Su-In Lee (2017)
