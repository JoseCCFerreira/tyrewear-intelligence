select
  tyre_id,
  measurement_date,
  td_initial_mm,
  td_current_mm,
  greatest(td_initial_mm - td_current_mm, 0) as calculated_wear_mm,
  wear_rate_mm_10000km,
  remaining_life_km,
  predicted_life_km,
  risk_class
from {{ ref('stg_td_measurements') }}
