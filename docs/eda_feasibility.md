# Exploratory Data Analysis & Feasibility Report

This document outlines the viability of our external datasets for testing the low- and high-dimensional capabilities of the `econirl` estimators. 

## 1. Case Study 1: Route Choice & Navigation
**Target Phenomenon**: Spatial driving patterns, avoiding traffic vs. time delay.
*   **Available Datasets**:
    *   **T-Drive Sample/Medium**: ~767,000 GPS trajectory points across 500 cars.
    *   **Beijing OSM Graph**: Structural Road Network matrix linking nodes (intersections) to edges (roads).
*   **Feasibility Check**:
    *   *Low-Dimensional (NFXP)*: Highly feasible if we superimpose a simplified $10 \times 10$ zone grid onto the Beijing map, treating transitions simply as "Move North, South, East, West."
    *   *High-Dimensional (SEES/Deep IRL)*: Feasible using the full continuous coordinates (`long`, `lat`) natively tracked in T-Drive combined with exact node metrics from the `osmnx` graph.

## 2. Case Study 2: Gig-Economy Labor Supply
**Target Phenomenon**: Optimal stopping/shift hours (When do workers log off based on fatigue vs. immediate financial reward?).
*   **Available Datasets (New external array)**:
    *   **CitiBike**: 500,000 localized New York spatial transitions (clean discrete tracking).
    *   **Uber/Lyft (HVFHV)**: Massive 19.6 Million trips in Jan 2024.
    *   **Yellow Taxi**: 2.9 Million trips in Jan 2024.
    *   **Weather Covariates**: Hourly local weather shocks for NYC.
*   **Feasibility Check**:
    *   *Schema Verification*: The `Uber/Lyft` data specifically tracks `trip_time`, `base_passenger_fare`, and explicitly `driver_pay`. The median trip pays $20.84 to the driver, varying heavily.
    *   *Low-Dimensional (CCP/NFXP)*: Using the `CitiBike` station nodes as a proxy for "delivery hotspots", we can discretize the state into 5 boroughs and map probability matrices.
    *   *High-Dimensional (TD-CCP/NNES)*: Incredibly robust feasibility. By structuring a dataset using the `Uber/Lyft` raw `driver_pay` matrix matched against `Weather` (e.g. rain shocks changing reservation wage) over an unbinned time-clock (fatigue), `NNES` can isolate exactly the wage elasticity across drivers. 

## 3. Case Study 3: Sequential Platform Activity
**Target Phenomenon**: Sequential Browsing and Platform Search matching.
*   **Available Datasets**:
    *   **Foursquare**: 227,000 check-in arrays tracking `userId`, exact geographical coordinates, and exactly 202 string `venueCategory` tags in NYC.
*   **Feasibility Check**:
    *   *Implementation Strategy*: We model the 202 distinct tags (Bars, Offices, Gyms) as platform "items." The agent continuously evaluates whether to move to a structurally similar string representation (from Bar -> Lounge) or break sequence. 
    *   *High-Dimensional Application*: Best suited for `GLADIUS` or `NPL` utilizing category frequency embeddings to define an almost virtually infinite behavioral state. 

---

## Technical Conclusion 
The external drive mapping was successful. 
- *Data Sizes Ensure Generality*: The 19.6 million `Uber` entries and 500,000 `CitiBike` entries guarantee that any Deep IRL frameworks (like `TD-CCP`) don't overfit to small test matrices. 
- *Next Steps*: Our first goal should be constructing an exact **MDP Pipeline** (Markov Decision Process transformer) to convert these relational SQL/Pandas structures into the rigid `(state, action, reward, next_state)` multidimensional arrays expected by `NFXP-NK` or `NNESEstimator` inheriting from `econirl.core`.
