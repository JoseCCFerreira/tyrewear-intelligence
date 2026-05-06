select *
from {{ ref('sample_tyre_dataset') }}
where td_current_mm not between 0 and 12
   or td_initial_mm not between 5 and 12
   or wear_rate_mm_10000km < 0
   or remaining_life_km < 0
   or latitude not between -90 and 90
   or longitude not between -180 and 180
