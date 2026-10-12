-- 1. Build one row per customer.
-- This table contains history metadata only, not model features or labels.

DROP TABLE IF EXISTS customer_history_summary;

CREATE TABLE customer_history_summary
ENGINE=InnoDB
AS
SELECT
    customer_ID,
    COUNT(*) AS statement_count,
    MIN(S_2) AS first_statement_date,
    MAX(S_2) AS last_statement_date,
    DATEDIFF(MAX(S_2), MIN(S_2)) AS observation_span_days
FROM staging_monthly_raw
GROUP BY customer_ID;

-- 2. Add a primary key to support efficient joins.
ALTER TABLE customer_history_summary
    ADD PRIMARY KEY (customer_ID);

-- 3. Verify row count and history coverage.
SELECT
    COUNT(*) AS customer_count,
    MIN(statement_count) AS min_statements,
    ROUND(AVG(statement_count), 2) AS avg_statements,
    MAX(statement_count) AS max_statements,
    MIN(first_statement_date) AS earliest_first_statement,
    MAX(last_statement_date) AS latest_last_statement,
    ROUND(AVG(observation_span_days), 1) AS avg_observation_span_days
FROM customer_history_summary;

-- 4. Distribution of statement-history lengths.
SELECT
    statement_count,
    COUNT(*) AS customers
FROM customer_history_summary
GROUP BY statement_count
ORDER BY statement_count;

-- 5. Compare history length with target for descriptive analysis.
-- Keep target out of the feature table used for model training.
SELECT
    CASE
        WHEN h.statement_count <= 3 THEN '01: 1-3 statements'
        WHEN h.statement_count <= 6 THEN '02: 4-6 statements'
        WHEN h.statement_count <= 9 THEN '03: 7-9 statements'
        ELSE '04: 10+ statements'
    END AS history_group,
    COUNT(*) AS customers,
    SUM(l.target = 1) AS default_labels,
    ROUND(AVG(l.target) * 100, 2) AS labeled_default_pct
FROM customer_history_summary h
JOIN customer_labels l
    ON l.customer_ID = h.customer_ID
GROUP BY history_group
ORDER BY history_group;