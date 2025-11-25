Model Performance
We trained a Ridge Regression model (alpha=1) to predict user music preferences.
R² (Train/Test): ~0.46
Mean Squared Error: 257.01
Coefficient: -3.08
Cross-Validation Score: 0.46
Notes:
R² is moderate because user ratings are noisy and sparse—predicting music preferences is challenging.
Negative coefficients indicate some features may reduce predicted ratings, which can happen with inversely correlated song/user attributes.
Training and test scores are similar, showing no overfitting but some underfitting.
