# Football Match Predictor (XGBoost + Elo)

This project predicts the outcome of international football matches (`win`, `loss`, `draw`) using machine learning. It combines a custom Elo rating system, form tracking based on recent goal differences, and an XGBoost classifier with hyperparameter optimization.

## Features

- Encoded categorical features for teams and tournaments
- Elo rating calculation for each team
- Team form based on last 5 matches' goal difference
- Elo difference (`elo_diff`) as a feature
- XGBoost classifier with GridSearchCV for optimal parameters
- Class balancing using computed sample weights
- Confusion matrix and feature importance visualization
- Exports all trained trees to `xgb_trees.pdf`

## Classification Report

```
              precision    recall  f1-score   support

        draw       0.31      0.20      0.24      1634
        loss       0.54      0.67      0.60      2060
         win       0.69      0.71      0.70      3384

    accuracy                           0.58      7078
   macro avg       0.51      0.53      0.52      7078
weighted avg       0.56      0.58      0.57      7078
```

## Confusion Matrix

![Confusion Matrix](confusion_matrix.png)

## Feature Importance

![Feature Importance](feature_importance.png)

## File Overview

| File                      | Description                            |
|---------------------------|----------------------------------------|
| `vm_predictor.py`         | Main training pipeline                 |
| `evaluate.py`             | Runs simulations and model evaluation |
| `results.csv`             | Historical match data                 |
| `xgb_model.pkl`           | Trained XGBoost model                 |
| `confusion_matrix.png`    | Confusion matrix from test set        |
| `feature_importance.png`  | Visualized feature importance         |

## Requirements

```bash
pip install pandas xgboost scikit-learn matplotlib numpy
```

## Dataset

This dataset includes 47,960 results of international football matches starting from the very first official match in 1872 up to 2024. The matches range from FIFA World Cup to FIFI Wild Cup to regular friendly matches.

### Files

- `results.csv`: All historical match results  
- `shootouts.csv`: Matches with penalty shootouts  
- `goalscorers.csv`: Goal details per player  

> Note: Matches are strictly men’s full internationals and exclude Olympic Games and youth/B-team matches.

## Credits

Created by Oliver Ingvert (2025)