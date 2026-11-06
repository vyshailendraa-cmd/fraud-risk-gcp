-- Leakage-safe fraud features for PaySim (mobile money / e-wallet transactions).
--
-- step = hour of the simulation (1..743, about 31 days).
-- Behavioural features look ONLY at earlier hours (window frames end at
-- "1 PRECEDING" on step), so a transaction never sees itself, anything later,
-- or anything else in the same hour.
--
-- Features are computed over ALL transaction types (so a destination account's
-- full inflow history counts), then the model table keeps only TRANSFER and
-- CASH_OUT, the only types where fraud occurs in this data.
--
-- leak_* columns are balances AFTER the transaction. A real-time fraud check
-- runs BEFORE the money moves, so these would not exist yet. They are kept
-- only to demonstrate the leakage in train.py and must never be used as model
-- inputs.

CREATE OR REPLACE TABLE fraud.features AS
WITH enriched AS (
  SELECT
    t.*,

    -- Destination account ("mule") activity before this hour
    COUNT(*) OVER (
      PARTITION BY nameDest, dest_is_merchant ORDER BY step
      RANGE BETWEEN 24 PRECEDING AND 1 PRECEDING
    ) AS dest_txn_count_24h,
    IFNULL(SUM(amount) OVER (
      PARTITION BY nameDest, dest_is_merchant ORDER BY step
      RANGE BETWEEN 24 PRECEDING AND 1 PRECEDING
    ), 0) AS dest_amt_in_24h,
    COUNT(*) OVER (
      PARTITION BY nameDest, dest_is_merchant ORDER BY step
      RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS dest_txn_history,

    -- Sender activity before this hour
    COUNT(*) OVER (
      PARTITION BY nameOrig, orig_is_merchant ORDER BY step
      RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS orig_txn_history

  FROM fraud.raw_transactions AS t
)
SELECT
  txn_id,
  step,
  isFraud,
  isFlaggedFraud,               -- the existing rule system's flag (for comparison only)
  type,
  amount,
  LN(1 + amount) AS log_amt,
  MOD(step, 24) AS hour_of_day,

  -- Sender balance BEFORE the transaction (known at decision time)
  oldbalanceOrg,
  SAFE_DIVIDE(amount, oldbalanceOrg) AS amt_to_orig_balance,
  CAST(oldbalanceOrg > 0 AND amount >= oldbalanceOrg AS INT64) AS drains_orig_account,
  CAST(oldbalanceOrg = 0 AS INT64) AS orig_balance_zero,

  -- Receiver balance BEFORE the transaction
  oldbalanceDest,
  CAST(oldbalanceDest = 0 AS INT64) AS dest_balance_zero,

  dest_txn_count_24h,
  dest_amt_in_24h,
  dest_txn_history,
  CAST(dest_txn_history = 0 AS INT64) AS dest_first_seen,
  orig_txn_history,

  -- LEAKY: post-transaction balances (for the leakage demo only)
  newbalanceOrig AS leak_newbalanceOrig,
  newbalanceDest AS leak_newbalanceDest,
  oldbalanceOrg - amount - newbalanceOrig AS leak_orig_balance_error,
  oldbalanceDest + amount - newbalanceDest AS leak_dest_balance_error

FROM enriched
WHERE type IN ('TRANSFER', 'CASH_OUT');
