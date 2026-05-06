select
  country,
  region,
  avg(latitude) as latitude,
  avg(longitude) as longitude,
  count(*) as tyres,
  avg(td_current_mm) as td_avg_mm,
  avg(wear_rate_mm_10000km) as wear_rate_avg,
  avg(case when risk_class in ('high', 'critical') then 1 else 0 end) * 100 as risk_share_pct
from {{ ref('int_ml_features') }}
group by all
