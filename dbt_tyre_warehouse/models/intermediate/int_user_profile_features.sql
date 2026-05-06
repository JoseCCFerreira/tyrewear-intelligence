select
  country,
  region,
  vehicle_type,
  road_type,
  season_of_year,
  count(*) as profile_observations
from {{ ref('stg_user_profile') }}
group by all
