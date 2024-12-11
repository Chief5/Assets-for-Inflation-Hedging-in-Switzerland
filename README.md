# Inflation Hedging Strategies for Swiss Investors

![GitHub last commit](https://img.shields.io/github/last-commit/Chief5/Assets-for-Inflation-Hedging-in-Switzerland)
![License](https://img.shields.io/badge/license-MIT-blue)

This project investigates which asset classes provide a reliable hedge against inflation for Swiss investors. Using historical data, we analyze the performance of equities, bonds, real estate, and commodities during periods of high inflation. The findings are for informational and academic purposes only and should not be construed as financial advice or recommendations.

This paper investigates which asset classes provide a reliable hedge against inflation for Swiss investors. Using historical data, we analyze the performance of equities, bonds, real estate, and commodities during periods of high inflation. The findings presented in this project are for informational and academic purposes only and should not be construed as financial advice or recommendations. 

## Table of Contents

- [Overview](#overview)
- [Methodology](#methodology)
  - [Approaches](#approaches)
- [Data](#data)
- [Features](#features)
- [Installation](#installation)
- [License](#license)

## Overview

This project uses a combination of statistical techniques and dynamic portfolio optimization to determine which asset classes provide the best inflation hedge for Swiss investors. The analysis is implemented using Python and made accessible to all via Docker.

## Methodology

We employed three analytical approaches to evaluate asset performance:

### Approaches

1. **Rolling Window Analysis**  
   - Calculated the following metrics over rolling windows:
     - Regression Beta
     - Correlation Coefficient
     - Real Returns
   - Allows for tracking how these values evolve over time.

2. **Static Analysis**  
   - Calculated a single value (Beta, Correlation Coefficient, or Real Return) for the entire historical period to provide an overall perspective.

3. **Dynamic Analysis**  
   - Evaluated metrics (Beta, Correlation Coefficient, and Real Return) across multiple time horizons (2 years, 10 years, and maximum).  
   - Dynamically input different stocks to identify the optimal subset of stocks for portfolio performance based on time horizon.

## Data

- **Inflation Data**:  
  Monthly CPI data in CSV format.  

- **Asset Data**:  
  Historical performance data for equities, bonds, real estate, and commodities accesed via YahooFinance.

## Features

- **Three Analytical Approaches**:
  - Rolling window, static, and dynamic evaluations.
- **Dynamic Portfolio Optimization**:
  - Evaluate the best-performing subset of stocks based on specified metrics and time horizons.
- **Interactive Tool**:
  - Input different stocks and evaluate portfolio performance dynamically.
- **Dockerized Environment**:
  - Fully containerized for easy replication and accessibility.

## Installation

To replicate the environment and run the project, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/Chief5/Assets-for-Inflation-Hedging-in-Switzerland.git
   cd your-repo
   ```
3. Install Docker if not already installed. Refer to the Docker installation guide.
4. Build the Docker image:
   ```
   docker build -t inflation-hedge .
   docker run --rm -it inflation-hedge bash
    ```
   
## License
This project is licensed under the MIT License. See the LICENSE file for details.
