"""Calendar accounting and unit conversion for published policy endpoints."""

from __future__ import annotations


def endpoint_metrics(record: dict, baseline: dict, config: dict, *, carbon: bool) -> dict:
    conversion = config['status_quo_gross_revenue_busd'] / baseline['calendar_revenue']
    baseline_d = config['selling_horizon_model'] / baseline['M']
    pax_conversion = config['status_quo_passengers_bn'] / (baseline['sales'] / baseline_d)
    d = config['selling_horizon_model'] / record['M']
    fixed_cost = float(baseline['fixed_cost'])
    welfare = (record['welfare_before_fixed'] - fixed_cost) / d
    baseline_welfare = (baseline['welfare_before_fixed'] - fixed_cost) / baseline_d
    ticket = record['ticket_tax_payment'] / d * conversion
    vacancy = record['vacancy_tax_payment'] / d * conversion
    ticket_lost = baseline['ticket_tax_payment'] / baseline_d * conversion
    # Only carbon-endpoint policies have the residual per-flight charge imposed.
    tau_c = float(record.get('implied_tau_c_model_per_flight', 0.0)) if carbon else 0.0
    carbon_revenue = tau_c / d * conversion
    gross = record['gross_revenue'] / d * conversion
    return {
        'M': float(record['M']),
        'load_factor_percent': 100 * record['sold_fraction'],
        'vacancy_percent': 100 * record.get('vacancy', record.get('vacancy_departing')),
        'passengers_bn': record['sales'] / d * pax_conversion,
        'net_revenue_busd': gross - ticket - vacancy - carbon_revenue,
        'welfare_gain_busd': (welfare - baseline_welfare) * conversion,
        'carbon_tax_usd_per_tonne': tau_c * conversion * 1000 / config['status_quo_emissions_mt'],
        'carbon_revenue_busd': carbon_revenue,
        'vacancy_revenue_busd': vacancy,
        'ticket_revenue_busd': ticket,
        'fiscal_dividend_busd': ticket + vacancy + carbon_revenue - ticket_lost,
        'gross_revenue_busd': gross,
        'consumer_surplus_gain_busd': (
            record['consumer_surplus'] / d - baseline['consumer_surplus'] / baseline_d
        ) * conversion,
        'flight_frequency_change_percent': 100 * (record['M'] / baseline['M'] - 1),
        'profit_after_fixed_before_carbon_model': (
            record['gross_revenue'] - baseline['c'] * record['sales']
            - record['ticket_tax_payment'] - record['vacancy_tax_payment'] - fixed_cost
        ),
    }
