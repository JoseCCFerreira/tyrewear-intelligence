select
  tyre_id,
  country,
  region,
  city,
  latitude,
  longitude,
  climate_zone,
  road_type,
  season_of_year
from {{ ref('sample_tyre_dataset') }}
