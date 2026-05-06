# 06. Tread Depth Theory

Tread depth (TD) measures the remaining rubber groove depth in millimetres.

Core formulas:

```text
wear_mm = td_initial_mm - td_current_mm
wear_rate_mm_10000km = wear_mm / mileage_km * 10000
remaining_life_km = max(td_current_mm - 1.6, 0) / wear_rate_mm_10000km * 10000
cost_per_1000km = price_eur / predicted_life_km * 1000
```

The legal/safety threshold used in the MVP is 1.6 mm.
