select
  tyre_id,
  country,
  region,
  city,
  latitude,
  longitude,
  climate_zone,
  road_type,
  case when climate_zone in ('cold', 'continental') then 'higher_weather_risk' else 'standard_weather_risk' end as climate_risk_band
from {{ ref('stg_geography') }}
