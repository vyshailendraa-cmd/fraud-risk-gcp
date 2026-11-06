-- Leakage-safe feature table for fraud detection.
-- Every behavioural feature looks ONLY at a card's PAST transactions
-- (window frames end at "1 PRECEDING"), so nothing from the current
-- or future transactions leaks into the features. No feature uses the label.
--
-- TransactionDT = seconds from a fixed reference point (IEEE-CIS).

CREATE OR REPLACE TABLE fraud.features AS
WITH base AS (
  SELECT
    TransactionID,
    isFraud,
    TransactionDT,
    TransactionAmt,
    ProductCD,
    card4,
    card6,
    P_emaildomain,
    -- Proxy for a card/customer identity (no true card ID is provided)
    CONCAT(
      IFNULL(CAST(card1 AS STRING), 'na'), '_',
      IFNULL(CAST(card2 AS STRING), 'na'), '_',
      IFNULL(CAST(card3 AS STRING), 'na'), '_',
      IFNULL(CAST(card5 AS STRING), 'na'), '_',
      IFNULL(CAST(addr1 AS STRING), 'na')
    ) AS card_key,
    C1, C2, C5, C6, C9, C11, C13, C14,
    D1, D4, D10, D15
  FROM fraud.raw_transactions
)
SELECT
  b.*,

  -- Time features
  MOD(CAST(FLOOR(TransactionDT / 3600) AS INT64), 24) AS hour_of_day,
  MOD(CAST(FLOOR(TransactionDT / 86400) AS INT64), 7) AS day_of_week,
  LN(1 + TransactionAmt) AS log_amt,

  -- Velocity: how many transactions this card made BEFORE this one
  COUNT(*) OVER (
    PARTITION BY card_key ORDER BY TransactionDT
    RANGE BETWEEN 3600 PRECEDING AND 1 PRECEDING
  ) AS txn_count_1h,
  COUNT(*) OVER (
    PARTITION BY card_key ORDER BY TransactionDT
    RANGE BETWEEN 86400 PRECEDING AND 1 PRECEDING
  ) AS txn_count_24h,
  IFNULL(SUM(TransactionAmt) OVER (
    PARTITION BY card_key ORDER BY TransactionDT
    RANGE BETWEEN 86400 PRECEDING AND 1 PRECEDING
  ), 0) AS amt_sum_24h,

  -- Spending profile from the card's history only
  COUNT(*) OVER (
    PARTITION BY card_key ORDER BY TransactionDT
    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
  ) AS card_txn_history,
  SAFE_DIVIDE(
    TransactionAmt,
    AVG(TransactionAmt) OVER (
      PARTITION BY card_key ORDER BY TransactionDT
      ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    )
  ) AS amt_vs_card_avg,

  -- Seconds since the card's previous transaction (NULL for first-ever)
  TransactionDT - LAG(TransactionDT) OVER (
    PARTITION BY card_key ORDER BY TransactionDT
  ) AS secs_since_prev_txn

FROM base b;
