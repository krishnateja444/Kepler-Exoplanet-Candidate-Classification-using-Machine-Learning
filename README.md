# Kepler-Exoplanet-Candidate-Classification-using-Machine-Learning
# 🌌 Exoplanet Detection using Machine Learning — Beginner Level

A beginner-friendly Machine Learning project based on **NASA/Kepler Objects of Interest (KOI)** data.

The goal of this project is to build an end-to-end **binary classification pipeline** that classifies Kepler Objects of Interest as:

* ✅ **CONFIRMED** — likely/confirmed exoplanet
* ❌ **FALSE POSITIVE** — object identified as a false positive

> **Important:** This project does not detect exoplanets directly from raw Kepler light curves.
> Instead, it uses already-derived astrophysical measurements from the Kepler KOI catalog.

This project is designed as the **first stage of a larger Engineering Physics + Machine Learning project**, starting with fundamental ML concepts before progressing toward raw light-curve analysis and more advanced models.

---

## 🎯 Project Objectives

The main objectives are to understand and implement a complete beginner Machine Learning workflow:

1. Define a real-world scientific ML problem
2. Load and understand tabular astronomical data
3. Perform data cleaning
4. Perform Univariate and Bivariate EDA
5. Understand physical meaning of features
6. Perform physics-based feature engineering
7. Split data into training and testing sets
8. Establish a baseline model
9. Scale numerical features
10. Train a Logistic Regression classifier
11. Evaluate the model using multiple metrics
12. Interpret model coefficients from a physics perspective

---

## 🔬 Problem Statement

Astronomers use the **transit method** to identify possible exoplanets.

When a planet passes in front of its host star, the observed brightness of the star can decrease slightly. The Kepler mission identified many such objects, which were catalogued as **Kepler Objects of Interest (KOIs)**.

In this beginner project, we use measurements already derived from those observations to answer:

> **Can Machine Learning classify a Kepler Object of Interest as CONFIRMED or FALSE POSITIVE using its astrophysical properties?**

This makes the problem a **binary classification problem**.

---

## 📊 Dataset

The project uses the **Kepler Objects of Interest (KOI)** tabular dataset.

### Target Variable

`koi_disposition`

| Original Value   | ML Label |
| ---------------- | -------: |
| `CONFIRMED`      |      `1` |
| `FALSE POSITIVE` |      `0` |

For this beginner version:

* `CANDIDATE` objects are removed
* `NOT DISPOSITIONED` objects are removed

This converts the problem into a clean binary classification task.

---

## 🧪 Features Used

The initial feature set was intentionally kept small and physically interpretable.

| Feature         | Description                         | Unit          |
| --------------- | ----------------------------------- | ------------- |
| `koi_period`    | Orbital/transit period              | Days          |
| `koi_duration`  | Transit duration                    | Hours         |
| `koi_depth`     | Transit depth                       | ppm           |
| `koi_model_snr` | Transit model signal-to-noise ratio | Dimensionless |
| `koi_steff`     | Stellar effective temperature       | K             |
| `koi_slogg`     | Stellar surface gravity             | log(cgs)      |
| `koi_srad`      | Stellar radius                      | Solar radii   |

The purpose of using these features is not simply to maximize model accuracy, but to understand **why these measurements might contain information about the classification**.

---

# 🔎 Exploratory Data Analysis

Before training the model, the dataset is explored to understand its structure and quality.

### EDA performed

#### 1. Class Distribution

We examine how many observations belong to each class.

This is important because an imbalanced dataset can make accuracy misleading.

#### 2. Numerical Summary

Descriptive statistics are calculated for the selected numerical features.

#### 3. Feature Distributions

Histograms are used to examine:

* central tendency
* spread
* skewness
* unusual distributions

#### 4. Outlier Analysis

Boxplots are used to identify extreme values in the astrophysical measurements.

#### 5. Feature vs Target Analysis

Boxplots are used to compare feature distributions between:

* FALSE POSITIVE
* CONFIRMED

#### 6. Correlation Analysis

A correlation heatmap is used to investigate relationships between the physical features.

---

# 🧠 Feature Engineering

One of the main learning goals of this project is understanding how **domain knowledge can be converted into ML features**.

Three types of physics-based transformations were introduced.

### 1. Transit Depth Fraction

The original transit depth is measured in parts per million (ppm).

We calculate:

$$
\text{Transit Depth Fraction}
=
\frac{\text{koi\_depth}}{10^6}
$$

and:

$$
\text{Transit Depth Percent}
=
\frac{\text{koi\_depth}}{10^4}
$$

This makes the physical interpretation of transit depth easier.

---

### 2. Transit Period Fraction

The duration and orbital period have different units, so we normalize them:

$$
f_{transit}
=
\frac{\text{duration (hours)}}{24 \times \text{period (days)}}
$$

This gives a dimensionless quantity representing approximately what fraction of one orbital period the object spends in transit.

---

### 3. Log Transformation

Astronomical measurements can have strongly skewed distributions.

We therefore create:

$$
\log(1+x)
$$

transformations for:

* orbital period
* model signal-to-noise ratio

These transformed features can reduce the influence of extreme values while retaining zero-valued observations.

---

# 🏗️ Machine Learning Pipeline

The overall workflow is:

```text
Kepler KOI Dataset
        │
        ▼
Data Understanding
        │
        ▼
Target Filtering
        │
        ▼
Missing Value Removal
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Physics-Based Feature Engineering
        │
        ▼
Train / Test Split
        │
        ▼
Feature Scaling
        │
        ├──────────────► Majority-Class Baseline
        │
        ▼
Logistic Regression
        │
        ▼
Predictions
        │
        ▼
Model Evaluation
        │
        ▼
Physics-Based Interpretation
```

---

# 📏 Train/Test Split

The dataset is divided into:

* **80% Training Data**
* **20% Testing Data**

A fixed random state of `42` is used for reproducibility.

The split is **stratified**, which helps preserve approximately the same class proportions in both datasets.

---

# ⚖️ Baseline Model

Before using Machine Learning, a simple **majority-class baseline** is created.

The baseline predicts the most common class for every test observation.

This gives us a reference point:

> A Machine Learning model should provide meaningful improvement over a trivial prediction strategy.

---

# 📐 Feature Scaling

Because Logistic Regression is sensitive to feature scale, `StandardScaler` is applied.

The transformation is:

$$
z = \frac{x-\mu}{\sigma}
$$

where:

* \(x\) = original feature value
* \(\mu\) = training-set mean
* \(\sigma\) = training-set standard deviation

### Important

The scaler is:

```text
fit()      → Training data only
transform() → Training data
transform() → Test data
```

This prevents information from the test set from leaking into preprocessing.

---

# 🤖 Machine Learning Model

## Logistic Regression

The first ML model used is **Logistic Regression**.

It was selected because it is:

* Simple
* Fast
* Suitable for binary classification
* Easy to interpret
* Useful as a strong baseline
* Suitable for understanding feature coefficients and probabilities

The model outputs a probability associated with the `CONFIRMED` class.

---

# 📊 Model Evaluation

Accuracy alone is not sufficient for evaluating the classifier.

The following metrics are calculated:

### Accuracy

Fraction of predictions that are correct.

### Balanced Accuracy

Useful when class distributions are not perfectly balanced.

### Precision

Of the objects predicted as confirmed, how many were actually confirmed?

### Recall

Of the actual confirmed objects, how many were identified by the model?

### F1 Score

Harmonic mean of precision and recall.

### ROC-AUC

Measures the model's ability to distinguish between the two classes across different classification thresholds.

### Confusion Matrix

Shows:

```text
                 Predicted
              FP        Confirmed

Actual FP      TN            FP

Actual
Confirmed     FN             TP
```

### ROC Curve

The ROC curve is also plotted to visualize classification performance.

---

# 🧬 Physics Interpretability

An important part of this project is connecting Machine Learning with Physics.

Because the features are standardized before Logistic Regression, the model coefficients can be compared by magnitude.

### Interpretation

**Positive coefficient**

→ pushes the prediction toward `CONFIRMED`

**Negative coefficient**

→ pushes the prediction toward `FALSE POSITIVE`

**Larger absolute coefficient**

→ stronger influence within this standardized Logistic Regression model.

However:

> A large model coefficient should not automatically be interpreted as a physical causal relationship.

Machine Learning identifies statistical relationships; it does not by itself establish physical causality.

---

# 🛠️ Technologies Used

```text
Python
├── NumPy
├── Pandas
├── Matplotlib
├── Seaborn
└── Scikit-learn
```

### Machine Learning

* Train/Test Split
* StandardScaler
* Logistic Regression
* Classification Metrics
* ROC-AUC
* Confusion Matrix

### Data Analysis

* Data Cleaning
* Univariate EDA
* Bivariate EDA
* Correlation Analysis
* Outlier Analysis
* Feature Engineering

---

# 📁 Project Structure

```text
exoplanet-detection-ml/
│
├── detection-of-exoplanet-beginner-lvl1-checkpoint.ipynb
│
├── cumulative.csv
│
└── README.md
```

---

# 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/exoplanet-detection-ml.git
```

### 2. Move into the project directory

```bash
cd exoplanet-detection-ml
```

### 3. Install the required libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 5. Open

```text
detection-of-exoplanet-beginner-lvl1-checkpoint.ipynb
```

Make sure `cumulative.csv` is available in the expected path before running the notebook.

---

# 📚 What I Learned

This project helped build my understanding of the complete beginner Machine Learning workflow:

```text
Problem Definition
        ↓
Data Collection
        ↓
Data Understanding
        ↓
Data Cleaning
        ↓
EDA
        ↓
Feature Engineering
        ↓
Train/Test Split
        ↓
Scaling
        ↓
Baseline
        ↓
Model Training
        ↓
Evaluation
        ↓
Interpretation
```

The most important lesson from this project is that **Machine Learning is not just about training a model**.

A significant part of the work involves:

* understanding the problem
* understanding the data
* asking the right EDA questions
* engineering meaningful features
* preventing data leakage
* selecting appropriate evaluation metrics
* interpreting the results

---

# 🔭 Future Improvements

This project is intentionally the **Level 1 / Beginner stage**.

The next versions will gradually increase the Machine Learning and Physics depth.

## Level 2 — Classical Machine Learning

Planned improvements:

* Cross-validation
* Decision Trees
* Random Forest
* Gradient Boosting
* XGBoost
* Hyperparameter tuning
* Feature selection
* PCA
* Model comparison

---

## Level 3 — Explainable Machine Learning

Future interpretability techniques:

* SHAP
* Permutation importance
* Feature importance comparison
* Error analysis

---

## Level 4 — Real Exoplanet Detection

The major next step is moving away from pre-derived catalog features toward **raw Kepler light curves**.

This would involve:

```text
Raw Light Curve
       ↓
Signal Cleaning
       ↓
Noise Reduction
       ↓
Transit Detection
       ↓
Period Detection
       ↓
Phase Folding
       ↓
Feature Extraction
       ↓
Machine Learning / Deep Learning
       ↓
Exoplanet Classification
```

Potential future models include:

* CNNs
* Time-series models
* Deep Learning
* Uncertainty estimation

This would make the project significantly closer to an actual **automated exoplanet detection pipeline**.

---

# 🎓 Engineering Physics Connection

This project combines two areas:

### Physics

* Orbital period
* Transit duration
* Transit depth
* Stellar temperature
* Stellar radius
* Stellar surface gravity
* Signal-to-noise ratio

### Machine Learning

* Classification
* EDA
* Feature engineering
* Scaling
* Logistic Regression
* Model evaluation
* Model interpretation

This makes the project useful as a bridge between **Engineering Physics and Machine Learning**.

---

# ⚠️ Current Limitations

This beginner project has several important limitations.

1. It uses **catalog-level derived measurements**, not raw telescope light curves.
2. Only two target classes are considered.
3. Missing values are handled using simple row removal.
4. Only Logistic Regression is used as the primary ML model.
5. Cross-validation has not yet been implemented.
6. Hyperparameter optimization has not yet been performed.
7. The model should not be interpreted as a complete scientific exoplanet discovery system.

These limitations are intentional because this repository represents the **first stage of the project**.

---

# 🌟 Project Roadmap

```text
✅ Level 1 — Beginner
   ├── Data Cleaning
   ├── EDA
   ├── Feature Engineering
   ├── Baseline
   ├── Logistic Regression
   └── Evaluation

⬜ Level 2 — Intermediate ML
   ├── Cross Validation
   ├── Tree Models
   ├── XGBoost
   ├── Hyperparameter Tuning
   └── Model Comparison

⬜ Level 3 — Explainable ML
   ├── SHAP
   ├── Feature Importance
   └── Error Analysis

⬜ Level 4 — Advanced Astronomy + ML
   ├── Raw Light Curves
   ├── Signal Processing
   ├── Period Detection
   ├── Phase Folding
   └── Advanced Feature Engineering

⬜ Level 5 — Deep Learning
   ├── CNN
   ├── Time-Series Deep Learning
   └── Automated Exoplanet Detection
```

---

# 👨‍💻 Author

**Krishna Teja Thota**

Engineering Physics Student
Interested in:

* Machine Learning
* Data Science
* Physics-informed ML
* Scientific Computing

---

