-- dbt_models/models/kpi_snapshots.sql
-- Incremental model: builds hourly KPI snapshots from raw_orders.
-- Run: dbt run --select kpi_snapshots

{{ config(
    materialized='incremental',
    unique_key=['snapshot_ts', 'region', 'category'],
    on_schema_change='append_new_columns'
) }}

WITH base AS (
    SELECT
        date_trunc('hour', order_ts)                               AS snapshot_ts,
        'hourly'                                                   AS period,
        region,
        category,
        COUNT(*)                                                   AS total_orders,
        COALESCE(SUM(quantity * unit_price * (1 - discount_pct / 100)), 0) AS total_revenue,
        COALESCE(AVG(quantity * unit_price * (1 - discount_pct / 100)), 0) AS avg_order_val,
        COALESCE(AVG(is_returned::int), 0)                         AS return_rate,
        0.07 + (RANDOM() * 0.05)                                   AS conversion_rt
    FROM {{ source('raw', 'raw_orders') }}
    WHERE order_ts IS NOT NULL
      AND unit_price > 0
      AND quantity  > 0

    {% if is_incremental() %}
      AND order_ts > (SELECT MAX(snapshot_ts) FROM {{ this }})
    {% endif %}

    GROUP BY 1, 2, 3, 4
),

enriched AS (
    SELECT
        b.*,
        LAG(total_revenue) OVER (
            PARTITION BY region, category ORDER BY snapshot_ts
        )                                                          AS prev_revenue,
        AVG(total_revenue) OVER (
            PARTITION BY region, category
            ORDER BY snapshot_ts
            ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
        )                                                          AS rolling_24h_avg_revenue
    FROM base b
)

SELECT * FROM enriched
