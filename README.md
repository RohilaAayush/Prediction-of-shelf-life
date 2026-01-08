FMCG Shelf-Life Prediction Tool
An end-to-end Machine Learning project to predict the shelf life of FMCG products using synthetic data, Random Forest regression, and a Streamlit dashboard for interactive analysis and prediction.
---
📌 Project Overview
This project simulates real-world FMCG product data and builds a predictive system to estimate product shelf life based on key quality and storage parameters. It covers the full ML lifecycle — data generation, model training, evaluation, and deployment via a web app.

---

## 🧠 Key Features

- Synthetic FMCG product data generation
- Random Forest regression model
- Shelf-life prediction through Streamlit UI
- Feature importance visualization
- Exploratory data analysis dashboard
- Modular and easy-to-run project structure

---

## 📂 Project Structure
prediction of shelf life/
│
├── generate_data.py # Generate synthetic FMCG product data
├── product_data.csv # Generated dataset
├── train_model.py # Train ML model and save artifacts
├── app.py # Streamlit web application
├── model.pkl # Trained Random Forest model
├── encoder.pkl # Saved label encoder
├── requirements.txt # Project dependencies
└── README.md # Project documentation


---

## ⚙️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Matplotlib / Seaborn

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone <repository-url>
cd prediction-of-shelf-life

