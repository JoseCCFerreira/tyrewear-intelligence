select
  tyre_id,
  vehicle_type,
  vehicle_weight_kg,
  drive_type,
  wheel_position
from {{ ref('sample_tyre_dataset') }}
