# Cab Ride Price Prediction (Uber vs. Lyft)

This repository contains an end-to-end Machine Learning pipeline to predict the price of Uber and Lyft rides using historical tabular data. The project compares a baseline Linear Regression model against a hyperparameter-tuned **XGBoost Regressor**, demonstrating how tree-based gradient boosting excels at capturing the complex, non-linear relationships inherent in dynamic ride-share pricing.

## 📌 Project Objective
The goal is to estimate a continuous numerical value (the fare price) of a cab ride based on geographic distance, time of day, weather conditions, surge multipliers, and specific vehicle service tiers (e.g., UberX, Lyft XL). 

---

## 📊 Dataset Description
The model utilizes two datasets merged on geographic location and rounded hourly timestamps:
1. **Cab Rides Dataset**: Contains ride characteristics (`cab_type`, `name`, `price`, `distance`, `surge_multiplier`, etc.).
2. **Weather Dataset**: Provides environmental contexts (`rain`, `temperature`, etc.) that heavily influence dynamic surge pricing.

**Final Dataset Dimensions:** `(637,976 rows, 21 columns)`

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Data Manipulation:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`
* **Preprocessing & Modeling:** `scikit-learn` (`SimpleImputer`, `LabelEncoder`, `GridSearchCV`)
* **Gradient Boosting:** `XGBoost`

---

## 🚀 Pipeline Workflow

### 1. Data Merging & Temporal Alignment
* Timestamps are converted from Unix milliseconds to readable datetime formats.
* A custom `merge_date` compound key is constructed by mapping `location/source + hourly timestamp` to successfully merge weather data into the cab ride transactions.

### 2. Exploratory Data Analysis (EDA)
Key findings extracted during data exploration:
* **Trips by Hour:** Analyzed peak traffic volume patterns to pinpoint demand spikes.
* **Distance vs. Price:** Established variance and structural caps on base fares.
* **Weather Impact:** Proved dynamic surge impact by grouping average fares across `Raining` vs. `No Rain` categories, confirming higher baseline costs during rainfall.

### 3. Feature Engineering & Cleaning
* **Outlier Removal:** Trimmed the top 0.5% extreme outliers from both `price` and `distance` to prevent model distortion.
* **Temporal Features:** Extracted `day_of_week`, `is_weekend`, and `is_rush_hour`.
* **Missing Value Imputation:** Handled sparse weather metrics using a `Median` strategy imputation via `SimpleImputer`.
* **Distance Buckets:** Segmented continuous distances into categorical bins (`<0.5`, `0.5-2`, `2-5`, etc.) to ease pattern recognition.
* **Categorical Encoding:** Converted text metrics like `cab_type`, `product_id`, and `name` (vehicle tier) into numerical format using `LabelEncoder`.

### 4. Time-Based Validation Split
To prevent data leakage and mimic a realistic production setting, data was split using chronological dates rather than a random shuffle:
* **Train Set:** First 80% of unique chronological dates (`510,267` rows).
* **Test Set:** Last 20% of unique chronological dates (`113,357` rows).

---

## Model Training & Hyperparameter Tuning

1. **Baseline Model:** A `LinearRegression` model was trained to establish a baseline performance floor.
2. **Advanced Model:** An `XGBRegressor` was paired with `GridSearchCV` (3-fold Cross Validation optimizing for Negative Root Mean Squared Error).

### Best Hyperparameters Found:
```json
{
  "learning_rate": 0.1,
  "max_depth": 7,
  "n_estimators": 200
}
