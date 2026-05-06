select distinct
  country,
  region,
  vehicle_type,
  road_type,
  season_of_year
from {{ ref('sample_tyre_dataset') }}
