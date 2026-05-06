# 09. Machine Learning Methods

Classical ML is the correct MVP baseline for tabular tyre data.

Implemented:

- Ridge regression
- Random Forest regression
- Gradient Boosting regression
- Logistic Regression classification
- Random Forest classification

Tyre dimension is included in the model feature set:

- tyre_size as categorical data
- width_mm
- aspect_ratio
- rim_inch

This lets the model learn that wear and risk can differ by footprint, sidewall profile and wheel diameter.

Targets:

- remaining_life_km
- risk_class
- wear_rate_mm_10000km
- cost_per_1000km
- recommended_tyre_score
