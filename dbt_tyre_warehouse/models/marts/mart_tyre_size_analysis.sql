select
  tyre_size,
  width_mm,
  aspect_ratio,
  rim_inch,
  count(*) as tyres,
  avg(td_current_mm) as td_avg_mm,
  avg(wear_rate_mm_10000km) as wear_rate_avg,
  avg(remaining_life_km) as remaining_life_avg_km,
  avg(cost_per_1000km) as cost_per_1000km_avg,
  avg(case when risk_class in ('high', 'critical') then 1 else 0 end) * 100 as risk_share_pct
from {{ ref('int_ml_features') }}
group by all
