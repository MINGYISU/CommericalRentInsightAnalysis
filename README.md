# CommericalRentInsightAnalysis

This repository contains code and documentation for an analytical project exploring factors influencing real estate industry and commercial rent levels across California markets between 2018 and 2024. It also identifies how pandemic has impacted the commercial real estate sector.

## Project Overview

The goal of this project was to identify which property, tenant, and market-level features have the most significant impact on rent pricing in the commercial real estate sector. It also considered how the COVID-19 pandemic has affected these factors. The analysis was conducted using a combination of statistical modeling and machine learning techniques.

Using a large set of lease transaction data combined with external economic indicators (e.g., unemployment rates, occupancy trends), we analyzed how rent is shaped by a combination of:

- Market location and submarket dynamics
- Building class, size and region
- Leasing sector (e.g., legal, financial, tech)
- Temporal effects (e.g., COVID impact)
- Macro-level indicators (e.g., unemployment trends)

We also trained a feedforward neural network to predict rent levels based on these features, providing insights into the rent changes in the near future.

## Data Notice

 The data was obtained from a commercial real estate data provider and is subject to strict confidentiality agreements, making it **proprietary and restricted**. As such, **neither data files(or visualizations) nor concrete analysis results** are included in this repository. We only demostrate a high-level overview of the analysis process and coding techiques used.

## Tools and Technologies

- **Python**: The primary programming language used for data preprocess, visualizations and modeling.
  - Pandas: For data cleanings and manipulations
  - Matplotlib/Seaborn: For data visualizations
  - PyTorch: For feedforward neural network modeling (predicting rent levels in near future)
- **R**: For linear regression modeling and statistical analysis (Estimate, MSE, t-statistics, R-squared, F-statistics), capturing the impact of pandemic on rent level
