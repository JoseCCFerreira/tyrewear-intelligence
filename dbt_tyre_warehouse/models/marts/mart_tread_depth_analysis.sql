select
  brand,
  tyre_size,
  wheel_position,
  country,
  region,
  count(*) as tyres,
  avg(td_current_mm) as td_avg_mm,
  min(td_current_mm) as td_min_mm,
  avg(wear_rate_mm_10000km) as wear_rate_avg
from {{ ref('int_ml_features') }}
group by all
