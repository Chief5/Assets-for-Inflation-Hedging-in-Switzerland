import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.api import OLS, add_constant
import os

# Define asset tickers and dataset paths
gold = "GLD"
smi = "^SSMI"
sp500 = "^GSPC"

# Set file location
script_dir = os.path.dirname(os.path.abspath(__file__)) # Location of the script
project_root = os.path.abspath(os.path.join(script_dir, '..')) # One level up from the script's directory
dataset_path = os.path.join(project_root, 'Datasets', 'cpi_ch.csv') # Construct the path to the dataset

def pull_data(*tickers):
    # Create a dictionary to store the data for each ticker
    data_dict = {}
        
    # Loop through each ticker and download its data using download_ticker_data
    for ticker in tickers:
        data_dict[ticker] = download_ticker_data(ticker)
        
    # Combine all ticker data into a single DataFrame, aligning by date
    combined_data = pd.concat(data_dict, axis=1)
    return combined_data

def download_ticker_data(ticker):
        # Download maximum available data for the Gold
        asset = yf.download(ticker, interval="1mo", period="max")
        asset_adj_close = asset[['Adj Close']]
        # Reset the index to make 'Date' a regular column, if not already
        asset_adj_close = asset_adj_close.reset_index()

        # Rename columns to match the SMI format
        asset_adj_close.columns = ['Date', 'Value']

        # Convert 'Date' to datetime format and set as index (if not already a datetime type)
        asset_adj_close['Date'] = pd.to_datetime(asset_adj_close['Date'], errors='coerce')
        asset_adj_close.set_index('Date', inplace=True)

        asset_adj_close.index = asset_adj_close.index.tz_localize(None)

        # Sort by date in ascending order
        asset_adj_close = asset_adj_close.sort_index(ascending=True)
        return asset_adj_close

# Function to calculate quarterly returns for a single asset
def calculate_rates(df):
    # Loop through each asset column (ticker name only)
    for column in df.columns.get_level_values(0).unique():
        # Calculate Quarterly return rate
        df[f'{column}_Return_Rate_Quarterly'] = df[(column, 'Value')].pct_change(3) * 100  # 3 months (quarterly) change

    # Drop the second level in column names (if desired) after calculation
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    return df

# Function to calculate unexpected inflation using AR(1)
def calculate_unexpected_inflation(file_path):
    # Load inflation data
    cpi_data = pd.read_csv(file_path)
    cpi_data['Time'] = pd.to_datetime(cpi_data['Time'])
    cpi_data.set_index('Time', inplace=True)

    # Fit AR(1) model
    ar_model = ARIMA(cpi_data['Inflation (CPI)'], order=(1, 0, 0)).fit()
    cpi_data['Expected_Inflation'] = ar_model.fittedvalues
    cpi_data['Unexpected_Inflation'] = cpi_data['Inflation (CPI)'] - cpi_data['Expected_Inflation'] # Errors from AR represent unexpected inflation
    return cpi_data[['Expected_Inflation', 'Unexpected_Inflation']]

# Function to perform regression analysis and extract betas
def regression_analysis(asset_returns, inflation_data):
    # Merge asset returns with inflation data
    combined_data = pd.concat([asset_returns, inflation_data], axis=1).dropna()
    
    X = combined_data[['Expected_Inflation', 'Unexpected_Inflation']]
    y = combined_data.iloc[:, 0]  # First column contains the asset returns
    X = add_constant(X)  # Add constant for the regression

    model = OLS(y, X).fit()
    return model

# Main execution
if __name__ == "__main__":
    # Step 1: Download asset data and calculate quarterly returns
    asset_data = pull_data(gold, smi, sp500)
    quarterly_returns = calculate_quarterly_returns(asset_data)

    # Step 2: Load CPI data and calculate unexpected inflation
    inflation_data = calculate_unexpected_inflation(dataset_path)

    # Step 3: Perform regression for each asset and print betas
    for ticker in quarterly_returns.columns:
        print(f"\nRegression Analysis for {ticker}")
        model = regression_analysis(quarterly_returns[[ticker]], inflation_data)
        beta_1 = model.params['Expected_Inflation']
        beta_2 = model.params['Unexpected_Inflation']
        print(f"β1 (Expected Inflation): {beta_1:.4f}")
        print(f"β2 (Unexpected Inflation): {beta_2:.4f}")
        print(model.summary())
