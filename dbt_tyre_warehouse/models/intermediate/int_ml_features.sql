select
  c.*,
  m.measurement_date,
  m.mileage_km,
  m.td_current_mm,
  m.wear_rate_mm_10000km,
  m.remaining_life_km,
  m.risk_class,
  v.vehicle_type,
  v.vehicle_weight_kg,
  v.drive_type,
  v.wheel_position,
  g.country,
  g.region,
  g.city,
  g.latitude,
  g.longitude,
  g.climate_zone,
  g.road_type,
  g.season_of_year
from {{ ref('stg_tyre_catalog') }} c
join {{ ref('stg_td_measurements') }} m using (tyre_id)
join {{ ref('stg_vehicle') }} v using (tyre_id)
join {{ ref('stg_geography') }} g using (tyre_id)
