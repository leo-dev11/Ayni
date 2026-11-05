-- Agregados por cliente del historial de CUOTAS pagadas.
-- Entrada: vista `installments_payments` (13 605 401 filas, una por cuota).
-- Es la evidencia más directa de si el cliente paga tarde y cuánto paga de menos.

WITH por_cuota AS (
    SELECT
        SK_ID_CURR,
        SK_ID_PREV,
        -- días de atraso: positivo = pagó después del vencimiento, negativo = pagó antes
        DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT                    AS retraso_dias,
        AMT_PAYMENT                                             AS pagado,
        AMT_INSTALMENT                                          AS debido
    FROM installments_payments
)

SELECT
    SK_ID_CURR,

    count(*)                                                    AS inst_n_payments,
    count(DISTINCT SK_ID_PREV)                                  AS inst_n_contracts,

    -- ¿paga tarde?
    avg(CASE WHEN retraso_dias > 0 THEN 1.0 ELSE 0.0 END)::DOUBLE AS inst_late_rate,
    count(*) FILTER (WHERE retraso_dias > 0)                    AS inst_n_late,
    max(retraso_dias)                                           AS inst_max_delay_days,
    avg(GREATEST(retraso_dias, 0))                              AS inst_avg_delay_days,
    min(retraso_dias)                                           AS inst_min_delay_days,

    -- ¿paga completo?
    sum(pagado)                                                 AS inst_paid_sum,
    sum(debido)                                                 AS inst_due_sum,
    sum(pagado) / NULLIF(sum(debido), 0)                        AS inst_payment_ratio,
    avg(pagado - debido)                                        AS inst_avg_diff,
    min(pagado - debido)                                        AS inst_min_diff,
    count(*) FILTER (WHERE pagado < debido)                     AS inst_n_underpaid

FROM por_cuota
GROUP BY SK_ID_CURR
