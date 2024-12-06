import pandas as pd # Analyzing tabular data
import numpy as np # Numerical tools for mathematical op. and array handling
import matplotlib.pyplot as plt # Plots, visualising data
import yfinance as yf # Imports fin. data from Yahoo
import os # Handles file paths and directories for accessing datasets dynamically
import seaborn as sns # Data visualisation with statistical plotting
from sklearn.linear_model import LinearRegression
from itertools import chain, combinations # Generating combinations or subsets of data
from statsmodels.tsa.arima.model import ARIMA # Model TS to estimate exp. and unexp. inf.
from statsmodels.api import OLS, add_constant


# Asset tickers
#STOCK
smi = "^SSMI"
spi_etf = "CHSPI"
sp500 = "^GSPC"
world_etf= "URTH"
europe_etf= "IEUR"
em_etf ="EEM"

#COMMODITIES
gold = "GLD"
gold_etf = "RING"
gold_ch = "XAUCHF=X"

#FIXED INCOME
ten_year_gov_bond = "CH10YT=RR"
ch_gov_bond = "AGGS.SW"
ch_corp_bond = "CHBBB.SW"
tips_bond = "TIP"
treasury_etf = "GOVT"
inflation_bond ="INWG.L"
emerg_mark_bond = "EMB"
prime_site = "SPSN.W"

#REAL ESTATE
ch_property_etf = "CHREIT.SW"
vang_real_est_etf = "VNQ"
dev_eur_prop_etf = "REXP.DE"

#CRYPTOCURRENCY
btc = "BTC-USD"
eth = "ETH-USD"
crypto_index = "BITW"

#HERE YOU CAN GET INTERVALLS
monthyl ="1mo"
quarterly = "3mo"


#HERE YOU CAN GET TIME HORIZON
one_year = "1y"
two_year = "2y"
five_year = "5y"
ten_year = "10y"
max_year = "max"

# Set file location
script_dir = os.path.dirname(os.path.abspath(__file__)) # Location of the script
project_root = os.path.abspath(os.path.join(script_dir, '..')) # One level up from the script's directory
dataset_path = os.path.join(project_root, 'Digital Tools for Finance', 'Switzerland_inflation_cpi_govt.xlsx') # Construct the path to the dataset
tempfile_path = os.path.join(project_root, 'Digital Tools for Finance') # Construct the path to the dataset

# Check if the file already exists and create a unique filename
counter = 1

inflation_data = pd.read_excel(dataset_path, decimal=",")
print(inflation_data.head())

def make(*x):
    portfolio_assets = list(x)  # Create a list from the input arguments
    print(f"Created portfolio with assets: {portfolio_assets}")
    return portfolio_assets

def calculate_average_returns(data_dict):
    # Initialize empty lists to store YoY and MoM DataFrames
    yoy_dfs = []
    mom_dfs = []

    # Loop through each DataFrame in the dictionary
    for asset, df in data_dict.items():
        # Find the dynamically named YoY and MoM columns
        yoy_column = f'{asset}_Return_Rate_YoY'
        mom_column = f'{asset}_Return_Rate_MoM'
 
        # Ensure the columns exist before appending; otherwise raises an error
        if yoy_column not in df.columns or mom_column not in df.columns:
            raise ValueError(f"DataFrame for '{asset}' is missing required columns: {yoy_column} or {mom_column}.")

        # Add the relevant columns to the lists
        yoy_dfs.append(df[yoy_column])
        mom_dfs.append(df[mom_column])

    # Concatenate all YoY and MoM columns -> compute averages row-wise using pd.concat and ocmputes averagaes row-wise with .mean(axis=1)
    yoy_combined = pd.concat(yoy_dfs, axis=1)
    mom_combined = pd.concat(mom_dfs, axis=1)

    # Calculate the averages across all columns (axis=1 means row-wise)
    average_yoy = yoy_combined.mean(axis=1)
    average_mom = mom_combined.mean(axis=1)

    # Create a new DataFrame with the calculated averages
    result_df = pd.DataFrame({
        'average_YoY_return': average_yoy,
        'average_MoM_return': average_mom
    })

    return result_df

def generate_portfolio(*args, interval, period): # Generates a portfolio's average return by combining and processing data for multiple assets
    pulled_data = {}
    for tick in args: # *args allows the function to accept any number of arguments (you can pass multiple asset ticker to make or generate portfolio)
        pulled_data[tick] = pull_data(tick, interval=monthyl, period=ten_year)
    # Apply the calculate_rates function to each DataFrame in the dictionary
    calculated_data_dict = {key: calculate_rates(df) for key, df in pulled_data.items()}
    average_table = calculate_average_returns(calculated_data_dict)
    
    return average_table

# Retrive historical data for multiple assets (tickers), processes it, and combines it into a single DataFrame
def pull_data(*tickers, interval, period):
    # Create a dictionary to store the data for each ticker
    data_dict = {}
        
    # Loop through each ticker and download its data using download_ticker_data
    for ticker in tickers:
        data_dict[ticker] = download_ticker_data(ticker, interval, period)
        
    # Combine all ticker data into a single DataFrame, aligning by date
    combined_data = pd.concat(data_dict, axis=1)
    return combined_data

def download_ticker_data(ticker, interval, period):
        # Download maximum available data for the Gold
        asset = yf.download(ticker, interval=monthyl, period=ten_year) # yf.download fetsches historical data for the specified ticker
        print(asset.head())
        asset_adj_close = asset[['Adj Close']]
        # Reset the index to make 'Date' a regular column, if not already
        asset_adj_close = asset_adj_close.reset_index() # Resets the index so that the original Date index becomes a regular column.

        # Rename columns to match the SMI format
        asset_adj_close.columns = ['Date', 'Value']

        # Convert 'Date' to datetime format and set as index (if not already a datetime type)
        asset_adj_close['Date'] = pd.to_datetime(asset_adj_close['Date'], errors='coerce')
        asset_adj_close.set_index('Date', inplace=True)

        asset_adj_close.index = asset_adj_close.index.tz_localize(None) # Removes timezone information from the Date index (if any)

        # Sort by date in ascending order
        asset_adj_close = asset_adj_close.sort_index(ascending=True) # Sorts the DataFrame in ascending order of date (earliest to latest)
        return asset_adj_close

# Calculate returns for each asset.
def calculate_rates(df):
    # Loop through each asset column (ticker name only)
    for column in df.columns.get_level_values(0).unique():
        # Calculate Year-over-Year (YoY) return rate
        df[f'{column}_Return_Rate_YoY'] = df[(column, 'Value')].pct_change(12) * 100  # 12-month (YoY) change
            
        # Calculate Month-over-Month (MoM) return rate
        df[f'{column}_Return_Rate_MoM'] = df[(column, 'Value')].pct_change() * 100  # 1-month (MoM) change

    # Drop the second level in column names (if desired) after calculation
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    return df

# Calculate Unexpected Inflation using AR(1)
def calculate_unexpected_inflation(inflation_df):
    """
    Calculate expected and unexpected inflation using AR(1) model.
    Expected inflation: AR(1) fitted values.
    Unexpected inflation: Residuals (errors) from AR(1).
    """
    inflation = inflation_df['Inflation']

    # Fit AR(1) model
    ar_model = ARIMA(inflation, order=(1, 0, 0)).fit()

    # Calculate expected inflation (fitted values) and unexpected inflation (residuals from AR(1))
    inflation_df['Expected_Inflation'] = ar_model.fittedvalues
    inflation_df['Unexpected_Inflation'] = inflation - inflation_df['Expected_Inflation']
    
    return inflation_df

# Perform OLS Regression
def regression_analysis(asset_returns, inflation_data):
    """
    Perform OLS regression for asset returns on expected and unexpected inflation.
    Returns the regression model and its coefficients.
    """
    # Merge asset returns with inflation data
    combined_data = pd.concat([asset_returns, inflation_data[['Expected_Inflation', 'Unexpected_Inflation']]], axis=1).dropna() #dropna() removes rows with missing values

    # Define independent variables (Expected and Unexpected Inflation) and dependent variable (Asset Returns)
    X = combined_data[['Expected_Inflation', 'Unexpected_Inflation']]
    y = combined_data['Return']  # Replace 'Return' with your actual column name for asset returns

    # Add a constant for the regression
    X = add_constant(X)

    # Perform OLS regression
    model = OLS(y, X).fit()

    # Return the regression model
    return model

# Load Inflation Data
inflation_data = pd.read_excel(dataset_path)
inflation_data['Date'] = pd.to_datetime(inflation_data['Date'], errors='coerce')
inflation_data.set_index('Date', inplace=True)

# Main Function to Analyze Hedging Capability
def analyze_hedging_capability(asset_ticker, inflation_data, interval="1mo", period="10y"):
    """
    Analyze an asset's ability to hedge inflation by assessing sensitivity to expected and unexpected inflation.
    """
    # Step 1: Pull asset data
    asset_data = download_ticker_data(asset_ticker, interval, period)
    
    # Step 2: Calculate asset returns
    asset_data = calculate_rates(asset_data)
    
    # Step 3: Calculate unexpected inflation -> adds exp. and unexp. inflation to the inf. DataFrame
    inflation_data = calculate_unexpected_inflation(inflation_data)

    # Step 4: Perform regression
    regression_result = regression_analysis(asset_data[['Return']], inflation_data)

    # Step 5: Output Results
    print(f"\nRegression Analysis for {asset_ticker}:")
    print(f"β1 (Expected Inflation): {regression_result.params['Expected_Inflation']:.4f}")
    print(f"β2 (Unexpected Inflation): {regression_result.params['Unexpected_Inflation']:.4f}")
    print(regression_result.summary())

    return regression_result

def generate_coefficients_table(asset_tickers, inflation_data, interval="1mo", period="10y"):
    """
    Generate a table with regression coefficients (β1, β2) for all asset classes.
    
    Parameters:
    asset_tickers (list): List of asset tickers.
    inflation_data (DataFrame): DataFrame containing inflation data with expected and unexpected inflation.
    interval (str): Data interval (e.g., "1mo").
    period (str): Time period for historical data (e.g., "10y").
    
    Returns:
    DataFrame: Table with regression coefficients for each asset.
    """
    results = []

    for ticker in asset_tickers:
        try:
            # Step 1: Pull asset data and calculate returns
            asset_data = download_ticker_data(ticker, interval, period)
            asset_data = calculate_rates(asset_data)
            
            # Step 2: Perform regression analysis
            model = regression_analysis(asset_data[['Return']], inflation_data)
            
            # Step 3: Store results
            results.append({
                "Asset": ticker,
                "β1 (Expected Inflation)": model.params['Expected_Inflation'],
                "β2 (Unexpected Inflation)": model.params['Unexpected_Inflation'],
                "R-squared": model.rsquared  # Goodness of fit for the regression
            })
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
    
    # Convert results to a DataFrame
    coefficients_table = pd.DataFrame(results)
    return coefficients_table

# Main Execution
if __name__ == "__main__":

    # Calculate expected and unexpected inflation
    inflation_data = calculate_unexpected_inflation(inflation_data)

    all_assets = [
        smi, spi_etf, sp500, world_etf, europe_etf, em_etf,
        gold, gold_etf, gold_ch, ten_year_gov_bond, ch_gov_bond,
        ch_corp_bond, tips_bond, treasury_etf, inflation_bond,
        emerg_mark_bond, ch_property_etf, vang_real_est_etf, dev_eur_prop_etf,
        btc, eth, crypto_index
    ]

    coefficients_table = generate_coefficients_table(all_assets, inflation_data)

    print("Regression Coefficients Table:")
    print(coefficients_table)