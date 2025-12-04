# 🎯 Customer Churn Prediction - Complete ML Project

## 📁 Project Structure
```
churn_prediction/
│
├── data/                          # Data folder
│   ├── raw_data.csv
│   ├── cleaned_data.csv
│   └── processed_data.csv
│
├── models/                        # Saved models
│   ├── trained_model.pkl
│   ├── final_model.pkl
│   └── scaler.pkl
│
├── 01_problem_definition.ipynb
├── 02_data_collection.ipynb
├── 03_Data_preprocessing.ipynb
├── 04_EDA.ipynb
├── 05_Feature_Engineering.ipynb
├── 06_Model_Selection.ipynb
├── 07_Model_Training.ipynb
├── 08_Model_Evaluation_Tuning.ipynb
├── 09_Deployment_App.py
└── 10_Monitoring.ipynb
```

## 🚀 Setup Instructions

### 1. Create Project Folders
```bash
mkdir churn_prediction
cd churn_prediction
mkdir data models
```

### 2. Install Required Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib streamlit
```

### 3. Run Notebooks in Order

#### Step 1: Problem Definition
- **File**: `01_problem_definition.ipynb`
- **What it does**: Defines the business problem and ML objectives
- **Output**: Problem documentation
- **Time**: 2 minutes

#### Step 2: Data Collection
- **File**: `02_data_collection.ipynb`
- **What it does**: Downloads and loads Telco Customer Churn dataset
- **Output**: `data/raw_data.csv`
- **Time**: 2 minutes

#### Step 3: Data Cleaning & Preprocessing
- **File**: `03_Data_preprocessing.ipynb`
- **What it does**: Handles missing values, removes duplicates, fixes data types
- **Output**: `data/cleaned_data.csv`
- **Time**: 3 minutes

#### Step 4: Exploratory Data Analysis (EDA)
- **File**: `04_EDA.ipynb`
- **What it does**: Analyzes patterns, creates visualizations, discovers insights
- **Output**: Charts and insights report
- **Time**: 5 minutes

#### Step 5: Feature Engineering
- **File**: `05_Feature_Engineering.ipynb`
- **What it does**: Creates new features, encodes categories, scales data
- **Output**: `data/processed_data.csv`, `models/scaler.pkl`
- **Time**: 4 minutes

#### Step 6: Model Selection
- **File**: `06_Model_Selection.ipynb`
- **What it does**: Trains 5 different models and compares performance
- **Output**: `data/best_model.txt`, comparison charts
- **Time**: 5 minutes

#### Step 7: Model Training
- **File**: `07_Model_Training.ipynb`
- **What it does**: Trains the best model with cross-validation
- **Output**: `models/trained_model.pkl`
- **Time**: 3 minutes

#### Step 8: Model Evaluation & Tuning
- **File**: `08_Model_Evaluation_Tuning.ipynb`
- **What it does**: Evaluates model, performs hyperparameter tuning
- **Output**: `models/final_model.pkl`, evaluation reports
- **Time**: 10 minutes (Grid Search takes time)

#### Step 9: Deployment (Web App)
- **File**: `09_Deployment_App.py`
- **What it does**: Creates interactive web application
- **How to run**:
```bash
streamlit run 09_Deployment_App.py
```
- **Access**: Opens at `http://localhost:8501`

#### Step 10: Monitoring
- **File**: `10_Monitoring.ipynb`
- **What it does**: Monitors model performance, detects drift, recommends retraining
- **Output**: Monitoring reports and alerts
- **Time**: 4 minutes

---

## 📊 Expected Results

### Model Performance:
- **Accuracy**: 85-87%
- **Precision**: 82-85%
- **Recall**: 80-83%
- **F1-Score**: 0.81-0.84

### Key Insights:
1. Month-to-month contracts have 43% churn rate
2. Electronic check payments have 45% churn rate
3. Customers without tech support have 42% churn rate
4. High monthly charges increase churn risk

---

## 🎓 What You'll Learn

### 1. Problem Definition
- Convert business problems to ML problems
- Define success metrics
- Identify data requirements

### 2. Data Collection
- Load data from various sources
- Check data quality
- Understand dataset structure

### 3. Data Preprocessing
- Handle missing values (fillna, dropna)
- Remove duplicates
- Convert data types
- Detect and handle outliers (IQR method)

**Formulas Used:**
- Missing % = (Missing values / Total) × 100
- IQR = Q3 - Q1
- Outlier: value < Q1 - 1.5×IQR or value > Q3 + 1.5×IQR

### 4. Exploratory Data Analysis
- Statistical analysis (mean, median, std, correlation)
- Visualizations (histograms, boxplots, heatmaps)
- Pattern discovery
- Feature-target relationships

**Formulas:**
- Correlation: r = Σ[(xi - x̄)(yi - ȳ)] / √[Σ(xi - x̄)² × Σ(yi - ȳ)²]
- Churn Rate = (Churned customers / Total customers) × 100

### 5. Feature Engineering
- Create new features (ChargesPerMonth, IsNewCustomer)
- Encode categorical variables (Label Encoding, One-Hot Encoding)
- Scale numerical features (StandardScaler)
- Feature selection (correlation-based)

**Formulas:**
- StandardScaler: z = (x - μ) / σ
- Label Encoding: Category → Number (Male=1, Female=0)
- One-Hot: Category → Binary columns

### 6. Model Selection
- Train multiple algorithms
- Compare performance metrics
- Select best model

**Models Used:**
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. SVM
5. XGBoost

### 7. Model Training
- Train-test split (80-20)
- Cross-validation (K-Fold, k=5)
- Fit model on training data

**Formulas:**
- Stratified Split: Maintains class distribution in train/test
- K-Fold CV: Average performance across k folds

### 8. Model Evaluation & Tuning
- Confusion Matrix (TP, TN, FP, FN)
- Classification metrics
- ROC-AUC curve
- Hyperparameter tuning (GridSearchCV)

**Formulas:**
- Accuracy = (TP + TN) / (TP + TN + FP + FN)
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
- ROC-AUC: Area under curve of TPR vs FPR

### 9. Deployment
- Save model (joblib/pickle)
- Create web interface (Streamlit)
- Real-time predictions
- User-friendly UI

### 10. Monitoring
- Track model performance over time
- Detect feature drift
- Alert system for degradation
- Retraining recommendations

**Monitoring Metrics:**
- Daily/Weekly accuracy trends
- False Negative Rate (critical for churn!)
- Feature distribution changes
- Prediction volume

---

## 🎯 Common ML Terms Explained

| Term | Simple Explanation | Example |
|------|-------------------|---------|
| **Features** | Input variables/columns | Age, Gender, Tenure |
| **Target** | What we want to predict | Churn (Yes/No) |
| **Training** | Teaching model with data | Model learns patterns |
| **Testing** | Checking model accuracy | Evaluate on unseen data |
| **Overfitting** | Model memorizes training data | 100% train, 60% test accuracy |
| **Underfitting** | Model too simple | 65% train, 64% test accuracy |
| **Cross-Validation** | Multiple train-test splits | 5-fold = 5 different splits |
| **Hyperparameters** | Model settings to tune | max_depth, learning_rate |
| **Pipeline** | Series of data processing steps | Clean → Transform → Train |

---

## 🔧 Troubleshooting

### Error: "No module named 'xgboost'"
```bash
pip install xgboost
```

### Error: "File not found: data/raw_data.csv"
- Make sure you run notebooks in order
- Cell 2 of notebook 02 downloads the data

### Error: "Model file not found"
- Complete notebooks 01-08 before running deployment
- Models are saved in steps 7 and 8

### Streamlit not opening?
```bash
# Check if streamlit is installed
pip show streamlit

# Run with full path
python -m streamlit run 09_Deployment_App.py
```

---

## 📚 Free Resources to Deploy

### 1. **Streamlit Cloud** (Recommended)
- **Website**: https://streamlit.io/cloud
- **Free Tier**: Yes
- **Setup**: 
  1. Push code to GitHub
  2. Connect Streamlit Cloud to repo
  3. Deploy in 1 click

### 2. **Render**
- **Website**: https://render.com
- **Free Tier**: 750 hours/month
- **Setup**: Connect GitHub and deploy

### 3. **Heroku**
- **Website**: https://heroku.com
- **Free Tier**: Limited (eco dyno)
- **Setup**: Git push to Heroku

### 4. **PythonAnywhere**
- **Website**: https://pythonanywhere.com
- **Free Tier**: 1 web app
- **Setup**: Upload files and configure

---

## 🎓 Next Steps to Learn More

1. **Try different datasets**: Credit card fraud, employee attrition
2. **Experiment with algorithms**: Neural Networks, LightGBM
3. **Add more features**: Create 10+ new engineered features
4. **Improve UI**: Add more charts to Streamlit app
5. **Deploy to cloud**: Make it accessible to everyone
6. **Learn deep learning**: TensorFlow, PyTorch
7. **Study MLOps**: Model versioning, CI/CD pipelines

---

## 📞 Support

If you face any issues:
1. Check if all libraries are installed
2. Verify you're running notebooks in order
3. Ensure all previous outputs exist
4. Read error messages carefully

---

## ✅ Completion Checklist

- [ ] Created project folders
- [ ] Installed all libraries
- [ ] Ran notebook 01 (Problem Definition)
- [ ] Ran notebook 02 (Data Collection)
- [ ] Ran notebook 03 (Data Preprocessing)
- [ ] Ran notebook 04 (EDA)
- [ ] Ran notebook 05 (Feature Engineering)
- [ ] Ran notebook 06 (Model Selection)
- [ ] Ran notebook 07 (Model Training)
- [ ] Ran notebook 08 (Evaluation & Tuning)
- [ ] Ran Streamlit app (09_Deployment_App.py)
- [ ] Ran notebook 10 (Monitoring)
- [ ] Tested predictions on new data
- [ ] Deployed to cloud (optional)

---

**Congratulations! 🎉 You've completed a full Machine Learning project from scratch!**

Total Time: ~45-60 minutes (excluding Grid Search)
Skill Level: Beginner to Intermediate
Real-world Application: Yes