


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import os
import seaborn as sns
from sklearn.linear_model import LinearRegression

#STOCK
nestle = "NESN.SW"
roche ="ROG.SW"
sp500 = "SPY"
smi = "^SSMI"

#COMMODITIES
gold = "GLD"

#FIXED INCOME
tips = "TIP"

#REAL ESTATE
ch_prime_site ="SPSN.SW"
eur_property_yield = "IPRP"
dow_jones_real_estate = "RWO"

#CRYPTOCURRENCY
btc = "BTC-USD"
eth = "ETH"

#HERE YOU CAN GET INTERVALLS

monthyl ="1mo"
quarterly = "3mo"


#HERE YOU CAN GET TIME HORIZON
one_year = "1y"
two_year = "2y"
five_year = "5y"
ten_year = "10y"
max_year = "max"

#set file location
script_dir = os.path.dirname(os.path.abspath(__file__))  # Location of the script
project_root = os.path.abspath(os.path.join(script_dir, '..'))  # One level up from the script's directory
dataset_path = os.path.join(project_root, 'Datasets', 'inflation_ch.csv') # Construct the path to the dataset


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
 
        # Ensure the columns exist before appending
        if yoy_column not in df.columns or mom_column not in df.columns:
            raise ValueError(f"DataFrame for '{asset}' is missing required columns: {yoy_column} or {mom_column}.")

        # Add the relevant columns to the lists
        yoy_dfs.append(df[yoy_column])
        mom_dfs.append(df[mom_column])

    

    # Concatenate all YoY and MoM columns
    yoy_combined = pd.concat(yoy_dfs, axis=1)
    mom_combined = pd.concat(mom_dfs, axis=1)

   
    # Calculate the averages across all columns (axis=1 means row-wise)
    average_yoy = yoy_combined.mean(axis=1)
    average_mom = mom_combined.mean(axis=1)

    # Create a new DataFrame with the calculated averages
    result_df = pd.DataFrame({
        'Average_YoY_return': average_yoy,
        'Average_MoM_return': average_mom
    })

    return result_df

def generate_portfolio(*args, interval, period):
    pulled_data = {}
    for tick in args:
        pulled_data[tick] = pull_data(tick, interval=interval, period=period)



    # Apply the calculate_rates function to each DataFrame in the dictionary
    calculated_data_dict = {key: calculate_rates(df) for key, df in pulled_data.items()}
    average_table = calculate_average_returns(calculated_data_dict)
    
    return average_table

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
        asset = yf.download(ticker, interval=interval, period=period)
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

def calculate_rates(df):
    # Loop through each asset column
    for column in df.columns.get_level_values(0).unique():
        # Calculate Year-over-Year (YoY) return rate
        df[f'{column}_Return_Rate_YoY'] = df[(column, 'Value')].pct_change(12) * 100  # 12-month (YoY) change
            
        # Calculate Month-over-Month (MoM) return rate
        df[f'{column}_Return_Rate_MoM'] = df[(column, 'Value')].pct_change() * 100  # 1-month (MoM) change




        # Calculate Year-over-Year (YoY) log return
    """         df[f'{column}_Return_Rate_YoY'] = (
            np.log(df[(column, 'Value')] / df[(column, 'Value')].shift(12))
        ) * 100  # Scaling for percentage interpretation
            
        # Calculate Month-over-Month (MoM) log return
        df[f'{column}_Return_Rate_MoM'] = (
            np.log(df[(column, 'Value')] / df[(column, 'Value')].shift(1))
        ) * 100  # Scaling for percentage interpretation
    """
    # Drop the second level in column names (if desired) after calculation
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    return df

# Helper function to calculate rolling beta
def calculate_rolling_beta(x, y, window):
    betas = []
    for i in range(len(x) - window + 1):
        x_window = x[i:i + window].values.reshape(-1, 1)
        y_window = y[i:i + window].values
        model = LinearRegression().fit(x_window, y_window)
        betas.append(model.coef_[0])
    return pd.Series(betas, index=x.index[window - 1:])

def calculate_single_beta(x, y):
    # Ensure x and y are properly aligned
    x_values = x.values.reshape(-1, 1) if hasattr(x, 'values') else np.array(x).reshape(-1, 1)
    y_values = y.values if hasattr(y, 'values') else np.array(y)
    
    # Fit the linear regression model
    model = LinearRegression().fit(x_values, y_values)
    
    # Return the beta (slope) coefficient
    return model.coef_[0]

def calculate_real_return(portfolio):
    #read dataset
    # Replace 'path_to_cpi_data.csv' with the actual path to your CPI CSV file
    cpi_data = pd.read_csv(dataset_path)

    #prepare dataset DATETIME and ORDER
    # Convert the 'Date' column to datetime format
    cpi_data['Date'] = pd.to_datetime(cpi_data['Date'], errors='coerce')
    cpi_data.set_index('Date', inplace=True)
    cpi_data = cpi_data.sort_index(ascending=True)
    cpi_data['Value'] = pd.to_numeric(cpi_data['Value'], errors='coerce')

    #get RATES
    cpi_data['Inflation_Rate_YoY'] = cpi_data['Value'].pct_change(12) * 100  # 12-month (YoY) change
    cpi_data['Inflation_Rate_MoM'] = cpi_data['Value'].pct_change() * 100  # 1-month (MoM) change


    """     #get RATES
    cpi_data['Inflation_Rate_YoY'] = (
        np.log(cpi_data['Value'] / cpi_data['Value'].shift(12)) * 100
    )  # 12-month YoY log return

    cpi_data['Inflation_Rate_MoM'] = (
        np.log(cpi_data['Value'] / cpi_data['Value'].shift(1)) * 100
    )  # 1-month MoM log return """


    #here we merge the dataset that same START and END point
    # - merge that same START and END point

    # Find the latest start date and earliest end date between the two datasets
    start_date = max(cpi_data.index.min(), portfolio.index.min())
    end_date = min(cpi_data.index.max(), portfolio.index.max())

    # Filter both datasets to only include this date range
    cpi_data = cpi_data[start_date:end_date]
    portfolio = portfolio[start_date:end_date]

    # Prepare to merge CPI with all assets' MoM and YoY rates
    # Step 1: Merge CPI and assets data on Date for Month-over-Month Rates
    merged_data_mom = pd.merge(
        cpi_data[['Inflation_Rate_MoM']], 
        portfolio.filter(like='_MoM_return'),  # Select all MoM rate columns
        left_index=True, 
        right_index=True
    )

    # Step 2: Merge CPI and assets data on Date for Year-over-Year Rates
    merged_data_yoy = pd.merge(
        cpi_data[['Inflation_Rate_YoY']], 
        portfolio.filter(like='YoY_return'),  # Select all YoY rate columns
        left_index=True, 
        right_index=True
    )

    # Step 2: Drop any rows with NaN values, as these will interfere with correlation calculation
    merged_data_mom.replace([np.inf, -np.inf], np.nan, inplace=True)
    merged_data_mom.dropna(inplace=True)

    merged_data_yoy.replace([np.inf, -np.inf], np.nan, inplace=True)
    merged_data_yoy.dropna(inplace=True)




    print(merged_data_mom)








    real_return_values = []

    # Calculate single real return for MoM (Month-over-Month) data
    for column in merged_data_mom.columns:






        if column != 'Inflation_Rate_MoM':  # Skip the CPI column itself
      
            # Calculate nominal return
            nominal_return = (1 + merged_data_mom[column] / 100).prod() - 1
            #inflation_adjustment = (1 + merged_data_mom['Inflation_Rate_MoM'] / 100).prod() - 1
            # Adjust for potential zero inflation rates during calculation
            inflation_rates = merged_data_mom['Inflation_Rate_MoM'].replace(0, 1e-6)
            inflation_adjustment = (1 + inflation_rates / 100).prod() - 1
            real_return = ((1 + nominal_return) / (1 + inflation_adjustment) - 1) * 100

            real_return_values.append(real_return)
            



    # Calculate single real return for YoY (Year-over-Year) data
    for column in merged_data_yoy.columns:
        if column != 'Inflation_Rate_YoY':  # Skip the CPI column itself

            # Calculate nominal return
            nominal_return = (1 + merged_data_yoy[column] / 100).prod() - 1
            # Calculate inflation adjustment
            inflation_adjustment = (1 + merged_data_yoy['Inflation_Rate_YoY'] / 100).prod() - 1
            # Calculate single real return
            real_return = ((1 + nominal_return) / (1 + inflation_adjustment) - 1) * 100
            print(f"Single YoY real return for {column} is {real_return:.4f}%")
            real_return_values.append(real_return)

    return real_return_values

def make_all_portfolios(asset_classes, intervals, time_horizons):
    all_portfolios_by_interval = {}

    # Loop through intervals to group portfolios
    for interval in intervals:
        # Initialize a dictionary for portfolios under this interval
        all_portfolios_by_interval[interval] = {}

        for asset_class in asset_classes:
            # Use the variable name (e.g., 'stocks', 'commodities') as the asset class name
            asset_class_name = [name for name in globals() if globals()[name] is asset_class][0]
            
            for time_horizon in time_horizons:
                # Generate a portfolio name
                portfolio_name = f"{asset_class_name}_{interval}_{time_horizon}"

                # Generate the portfolio for the current combination
                demo_portfolio = generate_portfolio(*asset_class, interval=interval, period=time_horizon)

                # Store the portfolio in the dictionary under the current interval
                all_portfolios_by_interval[interval][portfolio_name] = demo_portfolio

    return all_portfolios_by_interval

def calculate_real_return_for_all_portfolios(all_portfolios):
    # Initialize a dictionary to store the correlations
    correlations_by_interval = {}

    # Loop through intervals
    for interval, portfolios in all_portfolios.items():
        correlations_by_interval[interval] = {}
        
        # Loop through each portfolio
        for portfolio_name, portfolio_data in portfolios.items():
            # Calculate the correlation with CPI
            correlation = calculate_real_return(portfolio_data)
            
            # Store the correlation result
            correlations_by_interval[interval][portfolio_name] = correlation

    return correlations_by_interval

def plot_combined_correlation_table(correlations_dict):
    # Initialize lists to organize data for the combined table
    mom_data = []
    yoy_data = []
    intervals = []

    # Prepare data for each interval
    for interval, portfolios in correlations_dict.items():
        table_data_mom = {}
        table_data_yoy = {}

        for portfolio_name, correlation_values in portfolios.items():
            # Extract asset class and time horizon from the portfolio name
            parts = portfolio_name.split('_')
            asset_class = parts[0]
            time_horizon = parts[-1]

            # Add correlation values to respective dictionaries
            if time_horizon not in table_data_mom:
                table_data_mom[time_horizon] = {}
                table_data_yoy[time_horizon] = {}
            table_data_mom[time_horizon][asset_class] = round(correlation_values[0], 4)  # MoM Correlation
            table_data_yoy[time_horizon][asset_class] = round(correlation_values[1], 4)  # YoY Correlation

        # Convert to DataFrames
        table_df_mom = pd.DataFrame.from_dict(table_data_mom, orient='index')
        table_df_yoy = pd.DataFrame.from_dict(table_data_yoy, orient='index')

        # Append interval data
        mom_data.append((interval, table_df_mom))
        yoy_data.append((interval, table_df_yoy))

    # Plotting
    fig, axes = plt.subplots(nrows=len(mom_data), ncols=2, figsize=(16, len(mom_data) * 5))
    fig.suptitle("Real Return Analysis", fontsize=16)

    # Loop through intervals for MoM and YoY
    for i, (interval, mom_df) in enumerate(mom_data):
        yoy_df = yoy_data[i][1]  # Corresponding YoY DataFrame

        # Plot MoM table
        ax_mom = axes[i, 0] if len(mom_data) > 1 else axes[0]  # Adjust for single row
        ax_mom.axis('tight')
        ax_mom.axis('off')
        ax_mom.table(cellText=mom_df.values,
                     rowLabels=mom_df.index,
                     colLabels=mom_df.columns,
                     cellLoc='center', loc='center')
        ax_mom.set_title("MoM Real Return")

        # Plot YoY table
        ax_yoy = axes[i, 1] if len(mom_data) > 1 else axes[1]  # Adjust for single row
        ax_yoy.axis('tight')
        ax_yoy.axis('off')
        ax_yoy.table(cellText=yoy_df.values,
                     rowLabels=yoy_df.index,
                     colLabels=yoy_df.columns,
                     cellLoc='center', loc='center')
        ax_yoy.set_title("YoY Real Return")

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust space for the suptitle
    plt.show()

def summarize_dict(data_dict):
    summary = {}
    for key, value in data_dict.items():
        if isinstance(value, dict):  # If the value is a nested dictionary
            nested_summary = {nested_key: len(nested_value) for nested_key, nested_value in value.items()}
            summary[key] = {
                "type": "nested_dict",
                "num_keys": len(value),
                "lengths": nested_summary
            }
        else:  # If it's not a nested dictionary
            summary[key] = {
                "type": type(value).__name__,
                "length": len(value) if hasattr(value, '__len__') else None
            }
    return summary

#BUILD PORTFOLIOS HERE
stocks = make(smi)
commodities = make(gold)
fixed_income = make(tips)
real_estate = make(ch_prime_site, dow_jones_real_estate)
cryptocurrency = make(btc)

intervalls = [monthyl]
time_horizon = [two_year, five_year, ten_year, max_year]
asset_classes = [stocks, commodities, fixed_income, real_estate, cryptocurrency]



all_portfolios = make_all_portfolios(asset_classes, intervalls, time_horizon)
all_single_beta_values = calculate_real_return_for_all_portfolios(all_portfolios)

print("lenght of all portfolion" ,len(all_single_beta_values))
print("lenght of all portfolion" ,type(all_single_beta_values))

# Generate plots for the intervals
plot_combined_correlation_table(all_single_beta_values)
