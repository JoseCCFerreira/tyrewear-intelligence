select
  tyre_id,
  cast(measurement_date as date) as measurement_date,
  mileage_km,
  td_initial_mm,
  td_current_mm,
  td_inner_mm,
  td_center_mm,
  td_outer_mm,
  wear_mm,
  wear_rate_mm_10000km,
  predicted_life_km,
  remaining_life_km,
  cost_per_1000km,
  risk_class
from {{ ref('sample_tyre_dataset') }}
