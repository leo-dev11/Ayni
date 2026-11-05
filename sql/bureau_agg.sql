-- Agregados por cliente de los créditos que el solicitante tiene en OTRAS entidades.
-- Entrada: vista `bureau` (data/interim/bureau.parquet), 1 716 428 filas, varias por cliente.
-- Salida: una fila por SK_ID_CURR con la foto del historial externo.

SELECT
    SK_ID_CURR,

    -- volumen y estado del historial
    count(*)                                                      AS bureau_n_credits,
    count(*) FILTER (WHERE CREDIT_ACTIVE = 'Active')              AS bureau_n_active,
    count(*) FILTER (WHERE CREDIT_ACTIVE = 'Closed')              AS bureau_n_closed,
    count(DISTINCT CREDIT_TYPE)                                   AS bureau_n_types,

    -- deuda: lo prestado, lo que queda y lo vencido
    sum(AMT_CREDIT_SUM)                                           AS bureau_credit_sum,
    sum(AMT_CREDIT_SUM_DEBT)                                      AS bureau_debt_sum,
    sum(AMT_CREDIT_SUM_OVERDUE)                                   AS bureau_overdue_sum,
    sum(AMT_CREDIT_SUM_DEBT) / NULLIF(sum(AMT_CREDIT_SUM), 0)     AS bureau_debt_ratio,

    -- comportamiento de pago en el buró
    max(CREDIT_DAY_OVERDUE)                                       AS bureau_max_dpd,
    max(AMT_CREDIT_MAX_OVERDUE)                                   AS bureau_max_overdue_amt,
    count(*) FILTER (WHERE CREDIT_DAY_OVERDUE > 0)                AS bureau_n_delinquent,

    -- antigüedad: cuanto más negativo DAYS_CREDIT, más viejo el crédito
    avg(DAYS_CREDIT)                                              AS bureau_days_credit_mean,
    min(DAYS_CREDIT)                                              AS bureau_days_credit_min,
    max(DAYS_CREDIT_UPDATE)                                       AS bureau_last_update,

    -- mezcla de productos
    avg(CASE WHEN CREDIT_TYPE = 'Credit card' THEN 1.0 ELSE 0.0 END)::DOUBLE AS bureau_share_credit_card,
    avg(CASE WHEN CREDIT_TYPE = 'Consumer credit' THEN 1.0 ELSE 0.0 END)::DOUBLE AS bureau_share_consumer,
    sum(AMT_ANNUITY)                                              AS bureau_annuity_sum

FROM bureau
GROUP BY SK_ID_CURR
