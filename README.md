# Farmer-Income-Predictor
A machine learning project that predicts the income of Indian farmers based on various socio-economic and agricultural features. This model is built to aid policy-making, provide financial insights, and enable data-driven support for rural communities.
Farmer income predictor/
├── data/
│ ├── raw/
│ └── processed/
├── notebooks/
│ └── EDA_and_Modeling.ipynb
├── models/
│ └── trained_model.pkl
├── utils/
│ └── preprocessing.py
├── main.py
├── requirements.txt
└── README.md

markdown
Copy
Edit

## 📌 Problem Statement

To predict the income of farmers using features like landholding, crop type, livestock ownership, education, irrigation availability, etc. This prediction can support government schemes, NGO interventions, and agricultural credit risk assessments.

## 🧠 Technologies Used

- **Python 3.8+**
- **Pandas, NumPy** – Data manipulation
- **Matplotlib, Seaborn** – Data visualization
- **Scikit-learn** – ML model building
- **XGBoost / LightGBM** – Gradient boosting algorithms
- **Pickle** – Model serialization
- **Jupyter Notebook** – For exploratory analysis and experiments

## 🧪 Key Features

- Detailed EDA to uncover patterns in farmer data
- Feature engineering for improved prediction
- Trained multiple regression models with evaluation
- Optimized using metrics like RMSE, MAE, and R² score
- Final model saved and ready for deployment

## 📊 Results

- Achieved **R² Score**: *~0.87*
- Most important features:
  - Land size
  - Access to irrigation
  - Livestock count
  - Type of crops grown
  - Secondary sources of income

## 📁 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Farmer-income-predictor.git
   cd Farmer-income-predictor
