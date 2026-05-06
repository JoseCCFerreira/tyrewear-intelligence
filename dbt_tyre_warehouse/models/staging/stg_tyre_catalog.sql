select
  tyre_id,
  brand,
  model,
  tyre_size,
  width_mm,
  aspect_ratio,
  rim_inch,
  season_type,
  tyre_segment,
  fuel_efficiency_class,
  wet_grip_class,
  noise_db,
  rolling_resistance,
  price_eur
from {{ ref('sample_tyre_dataset') }}
