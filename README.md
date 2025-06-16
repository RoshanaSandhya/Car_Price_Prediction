## Car Price Prediction

An end-to-end machine learning project that predicts used car prices using the CarDekho dataset. Built with Python, scikit-learn, and Streamlit.

---

## 📂 Project Structure

```

car_price_project/
├── archive/                # Raw CSV files (e.g. Car details v3.csv)
├── models/                 # Saved artifacts:
│   ├── rf_model.pkl        # Trained Random Forest model
│   ├── scaler.pkl          # Fitted StandardScaler
│   └── feature_names.pkl   # List of feature names used by the model
├── notebooks/              # Jupyter notebooks (.ipynb) with code & analysis
│   └── Car_Sale_Prediction.ipynb
├── app.py                  # Streamlit web app to interactively predict prices
├── requirements.txt        # `pip install -r requirements.txt`
└── README.md               # This file

````

---

## 🚀 Setup & Installation

1. **Download or clone** this repository to your local machine.  
2. **Open a terminal** and navigate to the project root:
   ```bash
   cd path/to/car_price_project
    ````

3. **Create and activate** a virtual environment:

   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   # source venv/bin/activate # macOS / Linux
   ````
4. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

---

## 📓 Running the Notebook

1. With your `venv` activated, launch Jupyter:

   ```bash
   jupyter lab
   ```
2. Open **`notebooks/Car_Sale_Prediction.ipynb`** and run all cells in order:

   * Data ingestion & EDA
   * Preprocessing & feature engineering
   * Model training, evaluation & tuning
   * Summarize your findings

---

## 🌐 Launching the Streamlit App

1. Ensure the **models/** folder contains:

   * `rf_model.pkl`
   * `scaler.pkl`
   * `feature_names.pkl`
2. Run:

   ```bash
   streamlit run app.py
   ```
3. A browser window will open at `http://localhost:8501`.

   * Enter car details in the form
   * Click **Predict Price** to see your estimate

---

## 📝 Notes

* If you retrain the model or add features, re-export the feature list:

  ```python
  import pickle
  pickle.dump(
      X_train_scaled.columns.tolist(),
      open('models/feature_names.pkl','wb')
  )
  ```
* Always regenerate `requirements.txt` after adding new packages:

  ```bash
  pip freeze > requirements.txt
  ```

---

## 🤝 Acknowledgements

* **Dataset**: CarDekho on Kaggle
* **Author**: Buddana Roshana Sandhya
* **College**: NSRIT
