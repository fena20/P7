#!/usr/bin/env python3
"""
Phase 6: Scenario Enumeration and Economic Evaluation
Calculates NPV, viability scores, and tipping points across scenarios.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from itertools import product

from config import (
    OUTPUT_DIR, RANDOM_SEED, DISCOUNT_RATE, ANALYSIS_HORIZON,
    ENERGY_PRICES, HEAT_PUMP, RETROFIT_COSTS, CARBON_INTENSITY,
    VIABILITY_PARAMS, VIABILITY_THRESHOLD, CENSUS_DIVISIONS,
    ENVELOPE_CLASSES, CLIMATE_ZONES
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
np.random.seed(RANDOM_SEED)


def define_scenarios():
    """Define all scenario combinations."""
    
    # Retrofit options
    retrofits = list(RETROFIT_COSTS.keys())
    
    # Heat pump types
    hp_types = ['standard', 'cold_climate']
    
    # Climate zones (HDD ranges)
    climate_zones = {
        'Mild': 3000,
        'Moderate': 5000,
        'Cold': 7000
    }
    
    # Envelope classes
    envelope_classes = ['Poor', 'Medium', 'Good']
    
    # Price scenarios
    price_scenarios = ['low', 'base', 'high']
    
    # Generate combinations
    scenarios = []
    scenario_id = 0
    
    for retrofit in retrofits:
        for hp_type in hp_types:
            for climate, hdd in climate_zones.items():
                for envelope in envelope_classes:
                    for price in price_scenarios:
                        scenarios.append({
                            'scenario_id': scenario_id,
                            'retrofit': retrofit,
                            'hp_type': hp_type,
                            'climate_zone': climate,
                            'hdd': hdd,
                            'envelope_class': envelope,
                            'price_scenario': price
                        })
                        scenario_id += 1
    
    logger.info(f"Generated {len(scenarios)} scenarios")
    return pd.DataFrame(scenarios)


def calculate_baseline_energy(hdd, sqft=2000, envelope_score=0.5):
    """Calculate baseline gas consumption."""
    
    # Thermal intensity model (simplified)
    base_intensity = 0.008 - 0.004 * (envelope_score - 0.5)
    thermal_intensity = base_intensity * (1 + 0.0001 * (hdd - 5000))
    
    # Annual gas consumption (therms)
    gas_consumption = thermal_intensity * sqft * hdd / 100000
    
    return gas_consumption


def calculate_retrofit_savings(baseline_gas, retrofit_type, envelope_class):
    """Calculate energy savings from retrofit."""
    
    savings_factors = {
        'none': {'Poor': 0, 'Medium': 0, 'Good': 0},
        'air_sealing': {'Poor': 0.15, 'Medium': 0.10, 'Good': 0.05},
        'insulation_attic': {'Poor': 0.20, 'Medium': 0.12, 'Good': 0.05},
        'insulation_walls': {'Poor': 0.25, 'Medium': 0.15, 'Good': 0.08},
        'windows': {'Poor': 0.15, 'Medium': 0.10, 'Good': 0.05},
        'comprehensive': {'Poor': 0.40, 'Medium': 0.25, 'Good': 0.12}
    }
    
    factor = savings_factors.get(retrofit_type, {}).get(envelope_class, 0)
    savings = baseline_gas * factor
    
    return savings


def calculate_hp_energy(gas_consumption, cop):
    """Calculate heat pump electricity consumption."""
    
    # Convert gas (therms) to equivalent heat (kWh)
    # 1 therm = 29.3 kWh
    heat_demand_kwh = gas_consumption * 29.3
    
    # HP electricity consumption
    hp_electricity = heat_demand_kwh / cop
    
    return hp_electricity


def calculate_npv(scenario, sqft=2000):
    """Calculate 15-year NPV for a scenario."""
    
    # Extract parameters
    hdd = scenario['hdd']
    envelope_class = scenario['envelope_class']
    retrofit_type = scenario['retrofit']
    hp_type = scenario['hp_type']
    price_scenario = scenario['price_scenario']
    
    # Envelope score mapping
    envelope_scores = {'Poor': 0.3, 'Medium': 0.55, 'Good': 0.8}
    envelope_score = envelope_scores.get(envelope_class, 0.5)
    
    # Energy prices
    elec_price = ENERGY_PRICES['electricity'][price_scenario]
    gas_price = ENERGY_PRICES['natural_gas'][price_scenario]
    
    # HP parameters
    cop = HEAT_PUMP['COP_rated'] if hp_type == 'standard' else HEAT_PUMP['COP_17F'] + 0.5
    if hdd > 6000:  # Cold climate adjustment
        cop *= 0.85
    
    hp_cost = HEAT_PUMP['installation_cost'][hp_type]
    retrofit_cost = RETROFIT_COSTS[retrofit_type]
    
    # Baseline energy
    baseline_gas = calculate_baseline_energy(hdd, sqft, envelope_score)
    
    # Post-retrofit energy
    retrofit_savings = calculate_retrofit_savings(baseline_gas, retrofit_type, envelope_class)
    post_retrofit_gas = baseline_gas - retrofit_savings
    
    # HP energy consumption
    hp_electricity = calculate_hp_energy(post_retrofit_gas, cop)
    
    # Annual costs
    baseline_annual_cost = baseline_gas * gas_price
    hp_annual_cost = hp_electricity * elec_price
    annual_savings = baseline_annual_cost - hp_annual_cost
    
    # Emissions
    baseline_emissions = baseline_gas * CARBON_INTENSITY['natural_gas']
    hp_emissions = hp_electricity * CARBON_INTENSITY['electricity_2023']
    emissions_reduction = baseline_emissions - hp_emissions
    
    # NPV calculation
    total_investment = hp_cost + retrofit_cost
    npv = -total_investment
    
    for year in range(1, ANALYSIS_HORIZON + 1):
        npv += annual_savings / ((1 + DISCOUNT_RATE) ** year)
    
    # Simple payback
    if annual_savings > 0:
        payback_years = total_investment / annual_savings
    else:
        payback_years = 999
    
    return {
        'npv': npv,
        'annual_savings': annual_savings,
        'payback_years': payback_years,
        'baseline_gas_therms': baseline_gas,
        'hp_electricity_kwh': hp_electricity,
        'baseline_emissions_kg': baseline_emissions,
        'hp_emissions_kg': hp_emissions,
        'emissions_reduction_kg': emissions_reduction,
        'total_investment': total_investment
    }


def calculate_viability_score(hdd, elec_price, envelope_class):
    """Calculate HP viability score V."""
    
    alpha = VIABILITY_PARAMS['alpha']
    beta = VIABILITY_PARAMS['beta']
    gamma = VIABILITY_PARAMS['gamma'].get(envelope_class, 0.74)
    
    # Normalize inputs
    h_star = (hdd - 2000) / 6000  # 0-1 for HDD 2000-8000
    p_star = (elec_price - 0.08) / 0.14  # 0-1 for price 0.08-0.22
    
    # Clip to 0-1
    h_star = np.clip(h_star, 0, 1)
    p_star = np.clip(p_star, 0, 1)
    
    # Calculate V
    v = (1 - alpha * h_star) * (1 - beta * p_star) * gamma
    
    return v


def evaluate_all_scenarios(scenarios_df):
    """Evaluate NPV and viability for all scenarios."""
    
    logger.info("Evaluating all scenarios...")
    
    results = []
    
    for _, scenario in scenarios_df.iterrows():
        # Calculate NPV
        npv_results = calculate_npv(scenario)
        
        # Calculate viability
        elec_price = ENERGY_PRICES['electricity'][scenario['price_scenario']]
        viability = calculate_viability_score(
            scenario['hdd'],
            elec_price,
            scenario['envelope_class']
        )
        
        result = {**scenario.to_dict(), **npv_results, 'viability_score': viability}
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # Add viability classification
    results_df['is_viable'] = results_df['viability_score'] >= VIABILITY_THRESHOLD
    
    logger.info(f"Evaluated {len(results_df)} scenarios")
    logger.info(f"Viable scenarios: {results_df['is_viable'].sum()} ({results_df['is_viable'].mean()*100:.1f}%)")
    
    return results_df


def find_tipping_points(results_df):
    """Identify tipping points for HP adoption."""
    
    logger.info("Finding tipping points...")
    
    tipping_points = []
    
    # Group by envelope and climate
    for envelope in ['Poor', 'Medium', 'Good']:
        for climate in ['Mild', 'Moderate', 'Cold']:
            subset = results_df[
                (results_df['envelope_class'] == envelope) &
                (results_df['climate_zone'] == climate) &
                (results_df['retrofit'] == 'none')
            ]
            
            if len(subset) == 0:
                continue
            
            # Find price at which viability crosses 0.5
            viable = subset[subset['viability_score'] >= 0.5]
            
            if len(viable) > 0:
                min_viable_price = ENERGY_PRICES['electricity'][viable['price_scenario'].iloc[0]]
            else:
                min_viable_price = None
            
            # Find NPV break-even
            positive_npv = subset[subset['npv'] > 0]
            if len(positive_npv) > 0:
                break_even_scenario = positive_npv['price_scenario'].iloc[0]
            else:
                break_even_scenario = None
            
            tipping_points.append({
                'envelope_class': envelope,
                'climate_zone': climate,
                'mean_viability': subset['viability_score'].mean(),
                'mean_npv': subset['npv'].mean(),
                'min_viable_price': min_viable_price,
                'break_even_scenario': break_even_scenario,
                'pct_viable': (subset['viability_score'] >= 0.5).mean() * 100
            })
    
    return pd.DataFrame(tipping_points)


def calculate_division_summary(results_df):
    """Calculate summary statistics by Census division."""
    
    # Map climate zones to divisions (simplified)
    division_climate = {
        'New England': 'Cold',
        'Middle Atlantic': 'Cold',
        'East North Central': 'Cold',
        'West North Central': 'Cold',
        'South Atlantic': 'Mild',
        'East South Central': 'Moderate',
        'West South Central': 'Mild',
        'Mountain North': 'Cold',
        'Mountain South': 'Moderate',
        'Pacific': 'Moderate'
    }
    
    # Homes by division (millions)
    division_homes = {
        'New England': 2.1,
        'Middle Atlantic': 5.8,
        'East North Central': 7.2,
        'West North Central': 3.1,
        'South Atlantic': 4.5,
        'East South Central': 2.8,
        'West South Central': 3.2,
        'Mountain North': 1.5,
        'Mountain South': 1.4,
        'Pacific': 4.8
    }
    
    summary = []
    for division, climate in division_climate.items():
        subset = results_df[
            (results_df['climate_zone'] == climate) &
            (results_df['price_scenario'] == 'base')
        ]
        
        if len(subset) > 0:
            mean_v = subset['viability_score'].mean()
            mean_npv = subset['npv'].mean()
        else:
            mean_v = 0.5
            mean_npv = 0
        
        summary.append({
            'division': division,
            'abbr': [d['abbr'] for k, d in CENSUS_DIVISIONS.items() if d['name'] == division][0] if any(d['name'] == division for d in CENSUS_DIVISIONS.values()) else division[:3].upper(),
            'climate_zone': climate,
            'homes_millions': division_homes.get(division, 3.0),
            'mean_viability': mean_v,
            'mean_npv': mean_npv,
            'is_viable': mean_v >= 0.5
        })
    
    return pd.DataFrame(summary)


def save_scenario_results(scenarios_df, results_df, tipping_points, division_summary):
    """Save all scenario analysis results."""
    
    scenarios_df.to_csv(OUTPUT_DIR / "scenarios_defined.csv", index=False)
    results_df.to_csv(OUTPUT_DIR / "scenario_results.csv", index=False)
    tipping_points.to_csv(OUTPUT_DIR / "tipping_points.csv", index=False)
    division_summary.to_csv(OUTPUT_DIR / "division_summary.csv", index=False)
    
    logger.info(f"Saved scenario results to {OUTPUT_DIR}")


def main():
    """Execute Phase 6."""
    logger.info("=" * 60)
    logger.info("PHASE 6: SCENARIO ENUMERATION AND ECONOMIC EVALUATION")
    logger.info("=" * 60)
    
    # Define scenarios
    scenarios_df = define_scenarios()
    
    # Evaluate all scenarios
    results_df = evaluate_all_scenarios(scenarios_df)
    
    # Find tipping points
    tipping_points = find_tipping_points(results_df)
    logger.info(f"\nTipping Points:\n{tipping_points}")
    
    # Division summary
    division_summary = calculate_division_summary(results_df)
    logger.info(f"\nDivision Summary:\n{division_summary}")
    
    # Summary statistics
    logger.info("\n--- Summary Statistics ---")
    logger.info(f"Mean NPV: ${results_df['npv'].mean():,.0f}")
    logger.info(f"Mean Viability: {results_df['viability_score'].mean():.2f}")
    logger.info(f"Viable scenarios: {results_df['is_viable'].sum()} / {len(results_df)}")
    
    # Save results
    save_scenario_results(scenarios_df, results_df, tipping_points, division_summary)
    
    logger.info("\n✅ Phase 6 Complete")
    
    return results_df, tipping_points, division_summary


if __name__ == "__main__":
    main()
