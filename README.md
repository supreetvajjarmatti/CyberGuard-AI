# 🛡️ CyberGuard-AI
### Intelligent Network Intrusion Detection System Using Artificial Intelligence and Machine Learning

<p align="center">
  <b>An AI/ML-powered network intrusion detection system for identifying normal and malicious network traffic.</b>
</p>

---

## 📌 1. Project Overview

**CyberGuard-AI** is an Artificial Intelligence and Machine Learning-based Network Intrusion Detection System (NIDS) developed to identify potentially malicious network traffic.

The system uses supervised machine learning algorithms to classify network traffic into two categories: BENIGN and ATTACK.

It uses the CIC-IDS2017 dataset for model development and provides an interactive web dashboard using React.js, integrated with a FastAPI backend for network traffic prediction.

## 🎯 2. Project Objectives

- Develop an intelligent network intrusion detection system using Machine Learning.
- Identify normal and malicious network traffic.
- Detect potential DDoS attacks and other network threats.
- Train and compare multiple machine learning algorithms.
- Select an effective model for network traffic classification.
- Develop a web-based dashboard for displaying prediction results.
- Provide prediction confidence and threat risk levels.

## 🛠️ 3. Technologies Used

| Category | Technology | Purpose |
|---|---|---|
| Programming Language | Python | Machine learning development |
| Frontend | React.js | User interface |
| Frontend Build Tool | Vite | Frontend development |
| Styling | CSS | Dashboard design |
| Backend Framework | FastAPI | REST API development |
| ML Framework | Scikit-learn | Machine learning algorithms |
| ML Algorithm | XGBoost | Final classification model |
| Data Processing | Pandas | Dataset processing |
| Numerical Computing | NumPy | Numerical operations |
| Model Storage | Joblib | Saving trained models |
| CSV Processing | PapaParse | Reading CSV files |
| API Server | Uvicorn | Running FastAPI |
| Version Control | Git | Source code management |
| Repository Hosting | GitHub | Project hosting |

## 🤖 4. Machine Learning Algorithms

Four supervised machine learning algorithms are implemented and compared.

| Algorithm | Learning Technique | Purpose |
|---|---|---|
| Logistic Regression | Linear Classification | Baseline classification |
| Decision Tree | Tree-based Learning | Rule-based classification |
| Random Forest | Ensemble Learning | Improved classification |
| XGBoost | Gradient Boosting | Final intrusion detection |

**Selected Algorithm: XGBoost**

XGBoost is used as the final model for binary network traffic classification.

## 📊 5. Dataset Information

| Parameter | Description |
|---|---|
| Dataset Name | CIC-IDS2017 |
| Dataset Provider | Canadian Institute for Cybersecurity |
| Institution | University of New Brunswick |
| Total Records | 2,830,743 |
| Processed CSV Files | 8 |
| Input Features | 78 |
| Training Samples | 240,000 |
| Testing Samples | 60,000 |
| Total Sampled Records | 300,000 |
| Learning Type | Supervised Learning |
| Classification Type | Binary Classification |

**Dataset Source:**  
https://www.unb.ca/cic/datasets/ids-2017.html

### Classification Labels

| Target Value | Classification | Description |
|---|---|---|
| 0 | BENIGN | Normal network traffic |
| 1 | ATTACK | Potentially malicious network traffic |

## ⚙️ 6. Project Modules

| Module | Description |
|---|---|
| Data Collection | Collecting CIC-IDS2017 network traffic data |
| Data Preprocessing | Cleaning and preparing network traffic records |
| Feature Engineering | Preparing 78 numerical input features |
| Model Training | Training four machine learning algorithms |
| Model Evaluation | Comparing model performance |
| Model Selection | Selecting XGBoost for final prediction |
| Backend Development | Developing REST APIs using FastAPI |
| Frontend Development | Building the React dashboard |
| Threat Detection | Classifying network traffic as BENIGN or ATTACK |
| Result Visualization | Displaying prediction confidence and risk level |

## 📈 7. Model Performance

The following are the preliminary results obtained using a stratified sample of 300,000 records.

| Algorithm | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 91.22% | 70.38% | 95.74% | 81.12% |
| Decision Tree | 99.77% | 99.11% | 99.71% | 99.41% |
| Random Forest | 99.87% | 99.56% | 99.79% | 99.67% |
| XGBoost | **99.91%** | **99.66%** | **99.90%** | **99.78%** |

**Best-performing model in the preliminary evaluation: XGBoost**

*Note: These results are from a random stratified train-test split. Because network flow datasets can contain correlated records and duplicates, further validation using leakage-resistant splits is necessary before making general real-world performance claims.*

## 🏗️ 8. System Architecture

```text
             CIC-IDS2017 DATASET
                     |
                     v
             DATA PREPROCESSING
                     |
                     v
             FEATURE ENGINEERING
                     |
                     v
              MODEL TRAINING
                     |
          +----------+----------+
          |          |          |
          v          v          v
       Logistic   Decision   Random
      Regression   Tree      Forest
          |          |          |
          +----------+----------+
                     |
                     v
                  XGBOOST
                     |
                     v
               TRAINED MODEL
                     |
                     v
               FASTAPI BACKEND
                     |
                     v
               REACT DASHBOARD
                     |
                     v
             NETWORK TRAFFIC INPUT
                     |
                     v
              THREAT DETECTION
                     |
                     v
             BENIGN / ATTACK RESULT
```

## ✨ 9. Key Features

| Feature | Description |
|---|---|
| Machine Learning Detection | Identifies normal and malicious traffic |
| Multiple Algorithms | Compares four ML algorithms |
| XGBoost Classification | Uses the selected trained model |
| 78 Input Features | Processes network traffic attributes |
| CSV Upload | Allows users to upload network traffic datasets |
| Prediction API | Generates predictions through FastAPI |
| Interactive Dashboard | Displays results using React.js |
| Confidence Score | Displays predicted class probability |
| Risk Classification | Displays LOW, MEDIUM or HIGH risk |
| Threat Detection | Identifies BENIGN and ATTACK traffic |

## 🔌 10. API Endpoints

The FastAPI backend provides the following endpoints.

| HTTP Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API welcome message |
| GET | `/health` | Checks backend and model status |
| POST | `/predict` | Predicts network traffic classification |

### API Documentation

http://127.0.0.1:8000/docs

## 🧪 11. Testing Results

Initial functional testing was performed using normal network traffic and a confirmed DDoS-labelled traffic record.

| Test Case | Actual Traffic Label | Predicted Result | Confidence | Status |
|---|---|---|---:|---|
| Test 1 | BENIGN | BENIGN (0) | 99.99% | Passed |
| Test 2 | DDoS | ATTACK (1) | 99.98% | Passed |

*These are individual functional test results, not overall model accuracy.*

## 📁 12. Project Folder Structure

```text
CyberGuard-AI/
│
├── backend/
│   └── app/
│       ├── __init__.py
│       ├── main.py
│       ├── schemas.py
│       └── services/
│           ├── __init__.py
│           └── prediction.py
│
├── frontend/
│   ├── index.html
│   ├── package.json
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       └── App.css
│
├── ml/
│   ├── data/
│   │   ├── raw/
│   │   └── processed/
│   │
│   ├── preprocessing/
│   │   ├── clean_data.py
│   │   └── data_loader.py
│   │
│   ├── features/
│   │   └── feature_engineering.py
│   │
│   ├── training/
│   │   └── train_models.py
│   │
│   ├── evaluation/
│   │   ├── data_quality.py
│   │   └── evaluate_model.py
│   │
│   └── models/
│
├── database/
├── docs/
├── tests/
│
├── requirements.txt
├── .gitignore
└── README.md
```

## 💻 13. Installation and Execution

### Step 1: Clone the Repository

```bash
git clone https://github.com/supreetvajjarmatti/CyberGuard-AI.git

cd CyberGuard-AI
```

### Step 2: Create a Python Virtual Environment

```bash
python -m venv venv
```

Activate the environment in Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download the Dataset

Download CIC-IDS2017 from:

https://www.unb.ca/cic/datasets/ids-2017.html

Place the original CSV files inside:

```text
ml/data/raw/
```

### Step 5: Preprocess the Dataset

```bash
python ml/preprocessing/clean_data.py

python ml/features/feature_engineering.py
```

### Step 6: Train the Machine Learning Models

```bash
python ml/training/train_models.py
```

This generates the trained model files and feature-name information required by the backend.

### Step 7: Start the Backend

From the project root:

```bash
python -m uvicorn backend.app.main:app --reload
```

Backend URL:

http://127.0.0.1:8000

### Step 8: Start the Frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend URL:

http://localhost:5173/

## ⚠️ 14. Current Project Limitations

| Limitation | Details |
|---|---|
| Binary Classification | Currently identifies BENIGN and ATTACK only |
| CSV Analysis | Current dashboard analyzes only the first data row |
| Live Monitoring | Does not directly monitor live network packets |
| Model Validation | Further leakage-resistant validation is required |
| Explainability | SHAP-based explanations are planned |
| Database | Persistent prediction history is not yet implemented |

## 🔮 15. Future Enhancements

| Enhancement | Description |
|---|---|
| SHAP Explainability | Explain important features behind predictions |
| Multiclass Classification | Identify specific attack categories |
| Live Traffic Monitoring | Analyze live network traffic |
| Database Integration | Store prediction history |
| Advanced Dashboard | Add charts and analytics |
| Confusion Matrix | Visualize classification performance |
| ROC-AUC Analysis | Evaluate model discrimination |
| Automated Testing | Add backend and frontend test cases |

## 🎓 16. Academic Information

| Parameter | Details |
|---|---|
| Project Name | CyberGuard-AI |
| Project Domain | Artificial Intelligence and Machine Learning |
| Application Domain | Cybersecurity |
| Project Area | Network Intrusion Detection |
| Degree | M.Tech |
| Department | Computer Science and Engineering |
| Learning Approach | Supervised Machine Learning |
| Primary Algorithm | XGBoost |
| Dataset | CIC-IDS2017 |
| Frontend | React.js |
| Backend | FastAPI |
| Programming Language | Python |

## 👨‍💻 17. Developer Information

| Field | Details |
|---|---|
| Developer | Supreet Vajjaramatti |
| Degree | M.Tech |
| Specialization | Computer Science and Engineering |
| GitHub | https://github.com/supreetvajjarmatti |
| Repository | https://github.com/supreetvajjarmatti/CyberGuard-AI |

## 📜 18. License and Disclaimer

This project is developed for academic and educational purposes.

CyberGuard-AI is a machine learning research prototype and is not a replacement for a production-grade network security monitoring system.

---

⭐ **If you find this project interesting, consider giving the repository a star!**

**CyberGuard-AI — Making Networks Smarter and More Secure.**
