select
  tyre_id,
  fuel_efficiency_class,
  wet_grip_class,
  noise_db,
  rolling_resistance
from {{ ref('sample_tyre_dataset') }}
