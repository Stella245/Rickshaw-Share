# Rickshaw Share

A machine-learning-based rickshaw passenger matching system for Bangladesh.

## Models

- **Classification:** predicts whether two passengers are suitable for sharing (`match`: Match / Not Match).
- **Regression:** predicts expected extra detour distance in kilometres (`detour_km`). Regression excludes both `detour_km` and `match` from its inputs.

## Project structure

```text
data/rickshaw_matching_dataset.csv
models/best_rickshaw_classification_model.pkl
models/best_rickshaw_regression_model.pkl
train_models.py
predict.py
rickshaw_ml.ipynb
app.py
requirements.txt
```

## Installation and training

```bash
pip install -r requirements.txt
python train_models.py
```

The training script inspects the real CSV, performs EDA, compares four classification and four regression algorithms, uses leakage-safe pipelines, saves the best complete pipelines in `models/`, and runs a prediction example.

The notebook `rickshaw_ml.ipynb` is compatible with Jupyter Notebook and Google Colab. Run its cells after placing the project files in the working folder.

## Run the Streamlit application

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

The app collects passenger-pair details, runs classification first, and runs regression only when the pair is a suitable match. Saved sklearn pipelines include their own preprocessing, so the app does not scale inputs separately.

## Model flow

Passenger information → Classification → Match / Not Match → Regression → Estimated extra detour → Recommendation.
