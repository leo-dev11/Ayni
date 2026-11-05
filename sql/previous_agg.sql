-- Agregados por cliente de las solicitudes PREVIAS dentro de Home Credit.
-- Entrada: vista `previous_application` (1 670 214 filas).
-- Es el historial más informativo sobre el comportamiento del cliente con este mismo banco:
-- cuántas veces pidió, cuántas le rechazaron y por cuánto.

SELECT
    SK_ID_CURR,

    count(*)                                                                  AS prev_n_applications,
    count(DISTINCT NAME_CONTRACT_TYPE)                                        AS prev_n_contract_types,

    -- resultado de esas solicitudes
    avg(CASE WHEN NAME_CONTRACT_STATUS = 'Approved' THEN 1.0 ELSE 0.0 END)::DOUBLE AS prev_approval_rate,
    avg(CASE WHEN NAME_CONTRACT_STATUS = 'Refused'  THEN 1.0 ELSE 0.0 END)::DOUBLE AS prev_refusal_rate,
    count(*) FILTER (WHERE NAME_CONTRACT_STATUS = 'Refused')                  AS prev_n_refused,
    count(*) FILTER (WHERE NAME_CONTRACT_STATUS = 'Canceled')                 AS prev_n_canceled,

    -- montos solicitados vs. concedidos
    avg(AMT_APPLICATION)                                                      AS prev_amt_application_mean,
    max(AMT_APPLICATION)                                                      AS prev_amt_application_max,
    avg(AMT_CREDIT)                                                           AS prev_amt_credit_mean,
    avg(AMT_CREDIT / NULLIF(AMT_APPLICATION, 0))                              AS prev_credit_application_ratio,
    avg(AMT_ANNUITY)                                                          AS prev_annuity_mean,

    -- condiciones
    max(CNT_PAYMENT)                                                          AS prev_cnt_payment_max,
    avg(RATE_DOWN_PAYMENT)                                                    AS prev_down_payment_rate_mean,
    max(NFLAG_INSURED_ON_APPROVAL)                                            AS prev_insured_max,

    -- antigüedad: DAYS_DECISION negativo = cuántos días antes de la solicitud actual
    -- el más antiguo es el mínimo (más negativo); el más reciente, el máximo
    min(DAYS_DECISION)                                                        AS prev_days_first_decision,
    max(DAYS_DECISION)                                                        AS prev_days_last_decision,

    -- tipo de cliente: quien repite tiene historial, no es un desconocido
    avg(CASE WHEN NAME_CLIENT_TYPE = 'Repeater' THEN 1.0 ELSE 0.0 END)::DOUBLE AS prev_repeater_share

FROM previous_application
GROUP BY SK_ID_CURR
