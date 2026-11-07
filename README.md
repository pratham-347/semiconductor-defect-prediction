# Semiconductor Defect Prediction (SECOM) 

An end-to-end machine learning pipeline that detects defective semiconductor wafers on a manufacturing line. This project tackles extreme class imbalance and the "Curse of Dimensionality" in highly noisy sensor data, ultimately turning a black-box model into actionable hardware maintenance insights.

##  The Challenge
Modern semiconductor manufacturing relies on hundreds of sensors to monitor the assembly line. However:
1. **Extreme Noise:** Out of 590 sensors, the vast majority capture irrelevant noise rather than actual defect signals.
2. **Brutal Class Imbalance:** Defective chips are rare (roughly a 14:1 Pass-to-Fail ratio).
3. **Model Choking:** Standard tree-based models (Random Forest, XGBoost) completely failed to identify defects because they overfit on the hundreds of noisy features.

##  The Solution & Business Value
Instead of throwing complex models at bad data, I built a pipeline focused on **feature selection and signal extraction**:
* **Reduced dimensionality:** Used ANOVA (`SelectKBest`) to prove ~426 sensors were pure noise, stripping the dataset down to the **40 most critical features**.
* **Actionable Maintenance:** Extracted the logistic regression weights to identify the exact sensors (e.g., Sensor 436 and 431) causing defects, allowing hardware engineers to perform targeted machine recalibration.
* **Threshold Tuning:** Mapped out the financial trade-off for the factory manager—showing exactly how lowering the prediction threshold catches more defects at the cost of manageable manual inspection overhead.

## The Machine Learning Pipeline
1. **Data Preprocessing:** Imputed missing values (median) and dropped 116 constant features.
2. **Scaling:** Applied `StandardScaler` to normalize sensor readings.
3. **Feature Selection:** Grid-searched combinations of `k` to find the optimal 40 features using ANOVA statistical testing.
4. **Model Optimization:** Trained a `LogisticRegression` model with manual class weighting (`1:5`) and optimized regularization (`C=10.0`) via 5-fold Stratified Cross-Validation.

##  Final Performance (Test Set)
Evaluated on a completely unseen test set of 314 chips:
* **Accuracy:** 86.6%
* **Defects Caught (Recall at 0.50):** ~24%
* **False Alarms:** 26 (highly manageable for manual inspection)

### The Factory Trade-off
By adjusting the model's confidence threshold, the factory can choose its risk profile:
| Threshold | Defects Caught | Missed | False Alarms (Inspection Cost) |
|-----------|----------------|--------|--------------------------------|
| 0.50      | 5              | 16     | 26                             |
| 0.30      | 11             | 10     | 63                             |
| 0.10      | 17             | 4      | 166                            |

##  Repository Structure
```text
├── data/
│   └── secom.csv                   # The raw dataset
├── models/
│   └── secom_defect_model.pkl      # The final optimized pipeline (ready for deployment)
├── secom_defect_detection.ipynb    # Full exploratory data analysis and modeling notebook
└── README.md
