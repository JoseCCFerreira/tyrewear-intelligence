select
  m.tyre_id,
  m.td_current_mm,
  m.wear_rate_mm_10000km,
  m.remaining_life_km,
  c.price_eur / nullif(m.predicted_life_km, 0) * 1000 as cost_per_1000km,
  v.vehicle_weight_kg
from {{ ref('stg_td_measurements') }} m
join {{ ref('stg_tyre_catalog') }} c using (tyre_id)
join {{ ref('stg_vehicle') }} v using (tyre_id)
