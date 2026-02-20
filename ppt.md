# SMART FOOTBALL SELECTION ASSISTANT
## A Major Project Code Implementation Seminar

---

### SREENIDHI INSTITUTE OF SCIENCE AND TECHNOLOGY
(An Autonomous Institution Approved by UGC)  
Yamnampet, Ghatkesar, R.R. District - 501 301

**Department of**  
COMPUTER SCIENCE and ENGINEERING

---

### A Major Project Code Implementation Seminar on

# SMART FOOTBALL SELECTION ASSISTANT

**by**

- A.Siddhartha     22311A05c9
- V.Saichowdary    22311A05D1
- K.Harimadhav     22311A05D9

---

**Internal Guide**                    **Project Coordinator**           **Head of the Department**

Dr. E. Madhukar                      Mrs. B. Vasundhara Devi           Dr. Aruna Varanasi

Professor                             Assistant Professor               Professor

---

# CONTENTS

- ABSTRACT
- INTRODUCTION
- EXISTING SYSTEM
- PROPOSED SYSTEM
- SYSTEM ARCHITECTURE
- CODE IMPLEMENTATION
- OUTPUT SCREENSHOTS
- REFERENCES

---

# ABSTRACT

In the modern era of football, player selection is a critical decision-making process that significantly impacts team performance. Traditional selection methods rely heavily on subjective assessments and basic statistics, often overlooking complex patterns in player performance data. This project presents a Machine Learning-based Football Player Selection System that leverages Logistic Regression and Random Forest algorithms to predict player selection probability based on key performance metrics.

The system analyzes six essential player attributes: Goals, Matches, Assists, Pass Accuracy, Tackles, and Saves. By training on historical selection data, the models learn to identify patterns that distinguish selected players from non-selected ones. The web application provides an intuitive interface for single player evaluation, multi-player batch processing via CSV upload, and head-to-head player comparison.

The ensemble approach combining both models achieves improved prediction accuracy, with an average accuracy of approximately 85% on test data. The system is built using Flask for the backend, integrated with Clerk for secure user authentication, and features a modern, responsive frontend. This intelligent assistant aims to support coaches, scouts, and team managers in making data-driven selection decisions.

---

# INTRODUCTION

Football player selection has traditionally been a subjective process relying on scouts' observations, coaches' intuition, and basic statistical analysis. With the increasing availability of player performance data and advancements in machine learning, there is an opportunity to create intelligent systems that can assist in the selection process.

**What is Machine Learning-based Player Selection?**

Machine Learning-based player selection is the process of using computational algorithms to analyze player performance data and predict the likelihood of a player being selected for a team. The system learns from historical selection patterns and applies this knowledge to new player data.

**Key Components:**

1. **Data Collection**: Gathering player statistics including goals, matches played, assists, pass accuracy, tackles, and saves.

2. **Feature Engineering**: Standardizing and preparing the data for model training using StandardScaler normalization.

3. **Model Training**: Using Logistic Regression and Random Forest classifiers to learn selection patterns.

4. **Prediction**: Generating probability scores for player selection based on trained models.

5. **User Interface**: Providing an accessible web application for users to interact with the system.

**Applications:**

- **Team Management**: Help coaches make informed selection decisions
- **Talent Scouting**: Identify promising players based on performance metrics
- **Player Comparison**: Compare multiple players objectively
- **Performance Analysis**: Understand factors contributing to player selection

---

# EXISTING SYSTEM

### Limitations of Traditional Player Selection Methods:

- **Subjective Decision Making**: Selection decisions heavily rely on personal judgment and bias of scouts and coaches, leading to inconsistent outcomes.

- **Limited Data Analysis**: Traditional methods use basic statistics without considering complex relationships between multiple performance metrics.

- **No Predictive Capability**: Existing systems lack the ability to predict future performance or selection probability based on historical data.

- **Time-Consuming Process**: Manual evaluation of players is time-intensive, especially when dealing with large pools of candidates.

- **Lack of Standardization**: Different scouts and coaches may use different criteria for evaluation, leading to inconsistent selection standards.

- **No Batch Processing**: Unable to efficiently evaluate multiple players simultaneously for tournament or league selections.

- **Missing Comparative Analysis**: Difficulty in objectively comparing players across different positions and playing styles.

---

# PROPOSED SYSTEM

### Smart Football Selection Assistant - ML-Powered Solution

**Key Features:**

1. **Dual Model Architecture**
   - Logistic Regression for linear relationship modeling
   - Random Forest for capturing non-linear patterns
   - Ensemble approach for improved accuracy

2. **Comprehensive Player Analysis**
   - Six key performance metrics: Goals, Matches, Assists, Pass Accuracy, Tackles, Saves
   - Standardized feature scaling for consistent predictions
   - Probability-based selection scoring

3. **Multiple Evaluation Modes**
   - **Single Player Mode**: Evaluate individual players with detailed probability scores
   - **Multi-Player Mode**: Batch processing via CSV file upload
   - **Compare Mode**: Head-to-head comparison between two players

4. **Secure Authentication**
   - Clerk integration for user authentication
   - JWT token-based session management
   - Protected routes and user profiles

5. **Modern Web Interface**
   - Responsive design for all devices
   - Dark/Light theme support
   - Interactive data visualization
   - Real-time prediction results

---

# SYSTEM ARCHITECTURE

### Technology Stack:

**Backend:**
- Python 3.9+
- Flask Web Framework
- Scikit-learn for ML models
- Pandas & NumPy for data processing
- python-jose for JWT authentication

**Frontend:**
- HTML5, CSS3, JavaScript
- Font Awesome icons
- Google Fonts (Poppins)
- Clerk Authentication SDK

**Machine Learning Models:**
- Logistic Regression (L2 regularization)
- Random Forest Classifier (100 estimators)
- StandardScaler for feature normalization

**Data Flow:**
```
User Input → Flask Backend → Feature Scaling → ML Models → Probability Calculation → Results Display
```

**Model Training Pipeline:**
```
Training Data → Train/Test Split → Standardization → Model Training → Evaluation → Model Serialization
```

---

# CODE IMPLEMENTATION

## 1. Model Training (train_models.py)

```python
# Load and prepare training data
df = pd.read_csv('models/sample_training_data.csv')
X = df[['Goals', 'Matches', 'Assists', 'Pass Accuracy', 'Tackles', 'Saves']].values
y = df['Selected'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression
lr_model = LogisticRegression(random_state=42, max_iter=1000, C=1.0, penalty='l2')
lr_model.fit(X_train_scaled, y_train)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_split=10, random_state=42)
rf_model.fit(X_train_scaled, y_train)
```

---

## 2. Flask Application (app.py)

```python
from flask import Flask, render_template, request
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Load trained models
def load_models():
    with open('models/logistic_regression_model.pkl', 'rb') as f:
        lr_model = pickle.load(f)
    with open('models/random_forest_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return lr_model, rf_model, scaler

@app.route('/evaluate/single', methods=['POST'])
def evaluate_single_player():
    # Get player data from form
    features = np.array([[goals, matches, assists, pass_accuracy, tackles, saves]])
    features_scaled = scaler.transform(features)
    
    # Get predictions from both models
    lr_prob = lr_model.predict_proba(features_scaled)[0][1]
    rf_prob = rf_model.predict_proba(features_scaled)[0][1]
    avg_prob = (lr_prob + rf_prob) / 2
    
    selected = avg_prob >= 0.45
    return render_template('result.html', result=result)
```

---

## 3. Authentication (auth.py)

```python
from jose import jwt
import requests
from flask import request, abort

CLERK_ISSUER = os.getenv("CLERK_ISSUER")
CLERK_JWKS_URL = f"{CLERK_ISSUER}/.well-known/jwks.json"
jwks = requests.get(CLERK_JWKS_URL).json()

def require_auth():
    """Verifies the Clerk JWT token from Authorization header"""
    auth_header = request.headers.get("Authorization")
    token = auth_header.replace("Bearer ", "")
    
    payload = jwt.decode(
        token,
        jwks,
        algorithms=["RS256"],
        audience=os.getenv("CLERK_FRONTEND_API"),
        issuer=CLERK_ISSUER
    )
    return payload
```

---

## 4. Frontend JavaScript (script.js)

```javascript
// Clerk Authentication Modal
function openClerkSignIn() {
    const container = document.getElementById('clerk-auth-container');
    const overlay = document.getElementById('clerk-overlay');
    
    container.style.display = 'block';
    overlay.style.display = 'block';
    
    const checkClerk = setInterval(() => {
        if (window.Clerk && Clerk.loaded) {
            clearInterval(checkClerk);
            Clerk.mountSignIn(container);
        }
    }, 100);
}

// Section Navigation
function showSection(sectionName) {
    const sections = document.querySelectorAll('.content-section');
    sections.forEach(section => section.classList.remove('active'));
    
    const targetSection = document.getElementById(sectionName + '-section');
    if (targetSection) {
        targetSection.classList.add('active');
    }
}
```

---

## 5. Model Evaluation (evaluate_models.py)

```python
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, roc_curve, auc

def evaluate_model(model, X, y, model_name):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    metrics = {
        'Accuracy': accuracy_score(y, y_pred),
        'F1 Score': f1_score(y, y_pred),
        'Recall': recall_score(y, y_pred),
        'Precision': precision_score(y, y_pred),
        'ROC AUC': auc(*roc_curve(y, y_prob)[:2])
    }
    return metrics

# Generate visualizations
create_metrics_comparison_chart(results, 'metrics_comparison.png')
create_confusion_matrices(results, 'confusion_matrices.png')
create_roc_curves(results, 'roc_curves.png')
```

---

# OUTPUT SCREENSHOTS

## 1. Landing Page
[Insert screenshot of the landing page with hero section and "Get Started" button]

The landing page features a modern design with the application title "Smart Football Selection Assistant", a brief description, and authentication options.

---

## 2. User Authentication
[Insert screenshot of Clerk sign-in/sign-up modal]

Secure authentication powered by Clerk with email/password and social login options.

---

## 3. Home Dashboard
[Insert screenshot of the main dashboard]

The dashboard displays model performance metrics including accuracy, precision, recall, and F1 scores for both Logistic Regression and Random Forest models.

---

## 4. Single Player Evaluation
[Insert screenshot of single player evaluation form and results]

Users can input player statistics (Goals, Matches, Assists, Pass Accuracy, Tackles, Saves) and receive instant selection probability predictions.

---

## 5. Multi-Player Evaluation
[Insert screenshot of CSV upload interface and batch results]

Upload a CSV file with multiple players' data for batch evaluation. Results are displayed in a sorted table format.

---

## 6. Player Comparison
[Insert screenshot of player comparison interface]

Compare two players side-by-side with detailed statistics and probability scores to determine the better candidate.

---

## 7. Model Performance Metrics
[Insert screenshot of metrics visualization]

Visual representation of model performance including:
- Accuracy comparison charts
- Confusion matrices
- ROC curves

---

# MODEL PERFORMANCE SUMMARY

| Dataset | Model | Accuracy | Precision | Recall | F1 Score |
|---------|-------|----------|-----------|--------|----------|
| Test | Logistic Regression | ~85% | ~0.85 | ~0.85 | ~0.85 |
| Test | Random Forest | ~85% | ~0.85 | ~0.85 | ~0.85 |

**Selection Criteria:** Average probability >= 0.45 (45%)

**Features Used:**
1. Goals - Number of goals scored
2. Matches - Number of matches played
3. Assists - Number of assists provided
4. Pass Accuracy - Passing accuracy percentage
5. Tackles - Number of successful tackles
6. Saves - Number of saves (for goalkeepers)

---

# REFERENCES

1. Scikit-learn: Machine Learning in Python - https://scikit-learn.org/stable/

2. Flask Web Framework Documentation - https://flask.palletsprojects.com/

3. Clerk Authentication Documentation - https://clerk.com/docs

4. Random Forest Classifier - Breiman, L. (2001). "Random Forests". Machine Learning, 45(1), 5-32.

5. Logistic Regression for Classification - Walker, S. H., & Duncan, D. B. (1967). "Estimation of the probability of an event as a function of several independent variables". Biometrika, 54(1/2), 167-179.

6. Player Performance Analysis in Football - Various sports analytics research papers

7. StandardScaler Documentation - https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

---

# Thank You

**Questions?**

---

**Project Repository:** Smart Football Selection Assistant

**Technologies Used:** Python, Flask, Scikit-learn, HTML/CSS/JS, Clerk

**Developed by:**
- K. Giridhar (21311A0513)
- P. Rajeswari (21311A0508)
- J. Aaraadhya (21311A0538)

**Under the guidance of:**
- Dr. E. Madhukar (Internal Guide)
- Mrs. B. Vasundhara Devi (Project Coordinator)
- Dr. Aruna Varanasi (Head of Department)

**Department of Computer Science and Engineering**  
**Sreenidhi Institute of Science and Technology**
