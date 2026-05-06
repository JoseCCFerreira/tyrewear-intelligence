select
  tyre_id,
  brand,
  model,
  tyre_size,
  vehicle_type,
  country,
  region,
  remaining_life_km,
  risk_class,
  price_eur
from {{ ref('int_ml_features') }}
