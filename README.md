# Student Performance Predictor (AIML Minor Project)

## Overview
This project demonstrates a supervised machine-learning workflow that estimates a student's final marks using:
- Average study hours per day
- Attendance percentage
- Previous exam marks

A Random Forest Regressor is trained and evaluated, and a Streamlit interface lets a user enter values and view an estimated score.

## Features
- Sample student dataset in CSV format
- Train/test split for model evaluation
- Random Forest regression model
- Mean Absolute Error and R² evaluation metrics
- Interactive Streamlit prediction interface

## Project files
- `app.py` — Streamlit application and model training/evaluation
- `student_data.csv` — illustrative synthetic dataset
- `requirements.txt` — Python dependencies

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dataset note
The included 150-row dataset is synthetic and is provided only to demonstrate the project workflow. It is not real student data and should not be presented as collected institutional data. If your faculty requires a real dataset, replace it with an appropriately licensed, anonymized dataset and update this README.

## Limitations
Predictions are estimates based on a small illustrative dataset. Real student outcomes depend on many other factors. This project is for learning and demonstration, not for making academic decisions about students.

## Technologies
Python, Pandas, Scikit-learn, Streamlit.
