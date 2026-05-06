select
  tyre_id,
  tyre_size,
  width_mm,
  aspect_ratio,
  rim_inch,
  width_mm || '/' || aspect_ratio || ' R' || rim_inch as normalized_tyre_size
from {{ ref('stg_tyre_catalog') }}
