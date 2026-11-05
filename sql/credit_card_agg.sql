-- Agregados por cliente del uso de TARJETAS de crédito (saldos mes a mes).
-- Entrada: vista `credit_card_balance` (3 840 312 filas, una por tarjeta y mes).
-- El uso del límite es la señal clásica de tensión de liquidez.

SELECT
    SK_ID_CURR,

    count(*)                                                    AS cc_n_months,
    count(DISTINCT SK_ID_PREV)                                  AS cc_n_cards,

    -- uso del límite: saldo / límite
    avg(AMT_BALANCE / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0))::DOUBLE AS cc_utilization_mean,
    max(AMT_BALANCE / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0))::DOUBLE AS cc_utilization_max,

    -- tamaño del saldo
    avg(AMT_BALANCE)                                            AS cc_balance_mean,
    max(AMT_BALANCE)                                            AS cc_balance_max,
    avg(AMT_CREDIT_LIMIT_ACTUAL)                                AS cc_limit_mean,

    -- mora en la tarjeta
    max(SK_DPD)                                                 AS cc_max_dpd,
    avg(CASE WHEN SK_DPD > 0 THEN 1.0 ELSE 0.0 END)::DOUBLE     AS cc_delinquent_month_rate,

    -- actividad: disposiciones y pagos
    avg(AMT_DRAWINGS_CURRENT)                                   AS cc_drawings_mean,
    avg(CNT_DRAWINGS_CURRENT)                                   AS cc_n_drawings_mean,
    avg(AMT_PAYMENT_TOTAL_CURRENT)                              AS cc_payment_mean,
    avg(AMT_PAYMENT_TOTAL_CURRENT / NULLIF(AMT_TOTAL_RECEIVABLE, 0))::DOUBLE AS cc_payment_ratio

FROM credit_card_balance
GROUP BY SK_ID_CURR
