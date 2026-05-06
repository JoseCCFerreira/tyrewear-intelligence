select
  brand,
  tyre_segment,
  tyre_size,
  avg(remaining_life_km) as remaining_life_avg_km,
  avg(wear_rate_mm_10000km) as wear_rate_avg,
  avg(price_eur / nullif(remaining_life_km + mileage_km, 0) * 1000) as cost_per_1000km_avg
from {{ ref('int_ml_features') }}
group by all
