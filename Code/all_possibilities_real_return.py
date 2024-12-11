import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import yfinance as yf
import os
from sklearn.linear_model import LinearRegression
from itertools import chain, combinations
import tabulate as tab
# import seaborn as sns
# from PIL import Image, ImageDraw, ImageFont

#STOCKS
#Top 3 SMI Constituents by Market Capitalization
smi = "^SSMI"
roche = "ROG.SW"
nestle = "NESN.SW"
#Top 3 European Companies by Market Capitalization
novo_nordisk = "NVO"
lvmh = "MC.PA"
sap = "SAP"
#Top 3 S&P 500 Constituents by Market Capitalization
sp500 = "^GSPC"
apple = "AAPL"
nvidia = "NVDA"
#Top 3 Asian Companies by Market Capitalization
tsmc = "TSM"
tencent = "TCEHY"

#COMMODITIES
# Broad Commodity ETFs
invesco_commodity_composite_ucits_etf = "LGCF.L"
# Gold ETFs
ishares_physical_gold_etf = "IGLN.L"
# Energy ETFs
wisdomtree_brent_crude_oil = "BRNT.L"
# Agriculture ETFs
# Silver ETFs
ishares_physical_silver_etf = "ISLN.L"
# Specific Commodity ETFs
wisdomtree_natural_gas = "NGAS.L"
wisdomtree_wheat = "WEAT.L"
wisdomtree_corn = "CORN.L"
wisdomtree_soybeans = "SOYB.L"
# Leveraged and Inverse Commodity ETFs
# Commodity Equity ETFs
# Commodity Futures ETFs
# Commodity Currency-Hedged ETFs

#FIXED INCOME SECURITIES
# Broad Market Bond ETFs
ishares_global_corporate_bond_ucits_etf = "CORP.L"
# Government Bond ETFs
ishares_us_treasury_bond_7_10yr_ucits_etf = "IBTM.L"
# Corporate Bond ETFs
ishares_usd_corporate_bond_ucits_etf = "LQDE.L"
# High Yield Bond ETFs
ishares_euro_high_yield_corporate_bond_ucits_etf = "IHYG.L"
# Inflation-Linked Bond ETFs
ishares_euro_inflation_linked_govt_bond_ucits_etf = "IBCI.L"
ubs_etf_us_tips_ucits_etf = "TIPS.L"
# Short Duration Bond ETFs
ishares_euro_ultrashort_bond_ucits_etf = "ERNE.L"
# Emerging Markets Bond ETFs
ishares_jp_morgan_em_local_govt_bond_ucits_etf = "IEML.L"
# Corporate Bond ETFs by Maturity
# Aggregate Bond ETFs

#REAL ESTATE
# Swiss Real Estate Companies
swiss_prime_site = "SPSN.SW"
psp_swiss_property = "PSPN.SW"
allreal_holding = "ALLN.SW"
mobimo_holding = "MOBN.SW"
# Swiss Real Estate Funds
ubs_etf_sxi_real_estate = "SRECHA.SW"
procimmo_swiss_commercial_fund = "PSCF.SW"
# International Real Estate ETFs
ishares_us_real_estate_etf = "IYR"
ishares_global_reit_etf = "REET"

#CRYPTOCURRENCY
btc = "BTC-USD"
eth = "ETH-USD"
bnb = "BNB-USD"
xrp = "XRP-USD"
ada = "ADA-USD"

asset_class_map = {
  "stocks": [
        "^SSMI", "ROG.SW", "NESN.SW", 
        "NVO", "MC.PA", "SAP", 
        "^GSPC", "AAPL", "NVDA", 
        "TSM", "TCEHY"
    ],
    "commodities": [
        "LGCF.L", "IGLN.L", "BRNT.L", 
        "ISLN.L", "NGAS.L", "WEAT.L", "CORN.L", 
        "SOYB.L"
    ],
    "fixed_income": [
        "CORP.L", "IBTM.L", "LQDE.L", 
        "IHYG.L", "IBCI.L", "TIPS.L", 
        "ERNE.L", "IEML.L"
    ],
    "real_estate": [
        "SPSN.SW", "PSPN.SW", "ALLN.SW", "MOBN.SW", 
        "SRECHA.SW", "PSCF.SW", "IYR", "REET"
    ],
    "cryptocurrency": [
        "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD"
    ]
}

#HERE YOU CAN GET INTERVALS
monthly ="1mo"
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
tempfile_path = os.path.join(project_root, 'Datasets') # Construct the path to the dataset

# Check if the file already exists and create a unique filename
counter = 1

def make(*x):
    portfolio_assets = list(x)  # Create a list from the input arguments
    print(f"Created portfolio with assets: {portfolio_assets}")
    return portfolio_assets

def calculate_average_returns(data_dict, interval, period):
    # Initialize empty lists to store YoY and MoM DataFrames
    yoy_dfs = []
    mom_dfs = []

    # Loop through each DataFrame in the dictionary
    for asset, df in data_dict.items():
        # Find the dynamically named YoY and MoM columns
        yoy_column = f'{asset}_{interval}_{period}_YoY'
        mom_column = f'{asset}_{interval}_{period}_MoM'
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
        'average_YoY_return': average_yoy,
        'average_MoM_return': average_mom
    })

    return result_df

def generate_portfolio(*args, interval, period, data_table):
    column_suffix = f"{interval}_{period}"
    selected_data = {}
    # Extract relevant columns for each stock
    for stock in args:
        column_name = f"{stock}_{column_suffix}"
        if column_name in data_table.columns:
            selected_data[stock] = data_table[[column_name]]  # Select the specific column
        else:
            raise ValueError(f"Column '{column_name}' not found in the provided data_table.")
    
    # Apply the calculate_rates function to each DataFrame in the dictionary
    calculated_data_dict = {key: calculate_rates(df) for key, df in selected_data.items()}
    average_table = calculate_average_returns(calculated_data_dict, interval, period)


    return average_table

def pull_data(*tickers, intervals, periods):
    # Create a dictionary to store the data for each ticker
    data_dict = {}
        
    # Loop through each ticker and download its data using download_ticker_data
    for ticker in tickers[0]:
        for interval in intervals:
                for period in periods:
                        key = f"{ticker}_{interval}_{period}"
                        data_dict[key] = download_ticker_data(ticker, interval, period)
    
    # Combine all ticker data into a single DataFrame, aligning by date
    combined_data = pd.concat(data_dict, axis=1)
    #combined_data = pd.concat(data_dict.values(), axis=1, keys=data_dict.keys())
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
    # Loop through each asset column (ticker name only)
    for column in df.columns.get_level_values(0).unique():
        # Calculate Year-over-Year (YoY) return rate
        df[f'{column}_YoY'] = df[(column, 'Value')].pct_change(12) * 100  # 12-month (YoY) change
            
        # Calculate Month-over-Month (MoM) return rate
        df[f'{column}_MoM'] = df[(column, 'Value')].pct_change() * 100  # 1-month (MoM) change

    # Drop the second level in column names (if desired) after calculation
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    return df

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

def calculate_beta(portfolio, portfolio_name):
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

    real_return_values = []

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

    for column in merged_data_yoy.columns:
        if column != 'Inflation_Rate_YoY':  # Skip the CPI column itself
            # Calculate nominal return
            nominal_return = (1 + merged_data_yoy[column] / 100).prod() - 1
            # Calculate inflation adjustment
            inflation_adjustment = (1 + merged_data_yoy['Inflation_Rate_YoY'] / 100).prod() - 1
            # Calculate single real return
            real_return = ((1 + nominal_return) / (1 + inflation_adjustment) - 1) * 100
            # print(f"Single YoY real return for {column} is {real_return:.4f}%")
            real_return_values.append(real_return)


    return real_return_values

def make_all_portfolios(asset_classes, intervals, time_horizons, data_table):

    all_portfolios_by_interval = {}

    # Loop through intervals to group portfolios
    for interval in intervals:
        # Initialize a dictionary for portfolios under this interval
        all_portfolios_by_interval[interval] = {}

        for asset_class in asset_classes:
            # Use the variable name (e.g., 'stocks', 'commodities') as the asset class name
            #asset_class_name = [name for name in globals() if globals()[name] is asset_class][0]
            asset_class_name = "_".join(asset_class)

            for time_horizon in time_horizons:
                # Generate a portfolio name
                portfolio_name = f"{asset_class_name}_{interval}_{time_horizon}"

                # Generate the portfolio for the current combination
                demo_portfolio = generate_portfolio(*asset_class, interval=interval, period=time_horizon, data_table=data_table)

                # Store the portfolio in the dictionary under the current interval
                all_portfolios_by_interval[interval][portfolio_name] = demo_portfolio

    return all_portfolios_by_interval

def calculate_single_beta_for_all_portfolios(all_portfolios):

    # Initialize a dictionary to store the correlations
    correlations_by_interval = {}

    # Loop through intervals
    for interval, portfolios in all_portfolios.items():
        correlations_by_interval[interval] = {}
        
        # Loop through each portfolio
        for portfolio_name, portfolio_data in portfolios.items():
            # Calculate the correlation with CPI
            correlation = calculate_beta(portfolio_data, portfolio_name)
            
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
    fig.suptitle("Single Beta Analysis", fontsize=16)

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
        ax_mom.set_title("MoM Single Beta")

        # Plot YoY table
        ax_yoy = axes[i, 1] if len(mom_data) > 1 else axes[1]  # Adjust for single row
        ax_yoy.axis('tight')
        ax_yoy.axis('off')
        ax_yoy.table(cellText=yoy_df.values,
                     rowLabels=yoy_df.index,
                     colLabels=yoy_df.columns,
                     cellLoc='center', loc='center')
        ax_yoy.set_title("YoY Single Beta")

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

def make_data_table(*args, interval, period):

    data_table = pull_data(args, intervals=interval, periods=period)
    
    return data_table

def make_asset_class(*assets):

    asset_array = []

    for asset in assets:
        asset_array.append(asset)
    
    return asset_array

def find_subsets(array, min_size=1):
    if min_size > len(array):
        raise ValueError("min_size cannot be greater than the number of elements in the array.")
    
    # Generate all subsets of size >= min_size
    subsets = chain.from_iterable(
        combinations(array, r) for r in range(min_size, len(array) + 1)
    )
    subsets_as_lists = [list(subset) for subset in subsets]
    # Convert subsets to a list and return
    return subsets_as_lists

def make_all_portfolios_per_asset_class(subsets):
    all_portfolios = []
    for combination in subsets:
        all_portfolios.append(combination)
    return all_portfolios

def flatten_data(beta_values):

    data = []
    for interval, portfolios in beta_values.items():
        for portfolio_name, correlations in portfolios.items():
            data.append({
                "Interval": interval,
                "Portfolio": portfolio_name,
                "MoM Correlation": correlations[0],
                "YoY Correlation": correlations[1]
            })

    # Create DataFrame
    df = pd.DataFrame(data)

    return df

def split_and_sort(dataframe):

    # Split into MoM and YoY tables
    mom_table = dataframe[["Interval", "Portfolio", "MoM Correlation"]].sort_values(by="MoM Correlation", ascending=False)
    yoy_table = dataframe[["Interval", "Portfolio", "YoY Correlation"]].sort_values(by="YoY Correlation", ascending=False)

    # Reset indices for cleaner tables
    mom_table.reset_index(drop=True, inplace=True)
    yoy_table.reset_index(drop=True, inplace=True)

    return mom_table, yoy_table


def calcualte_beta_for_all(all_combinations):

    all_tables = {}
    for all in all_combinations:
        all_portfolios = make_all_portfolios(all, intervals, time_horizon, test_table)
        all_single_beta_values = calculate_single_beta_for_all_portfolios(all_portfolios)
        flattened_data = flatten_data(all_single_beta_values)
        mom_table, yoy_table = split_and_sort(flattened_data)
        # Determine the title based on asset class combinations
        title = "_".join(["_".join(combination) for combination in all])
       
        # Store the flattened and sorted tables in a dictionary with the title
        all_tables[title] = {
            #"flattened_data": flattened_data,
            "MoM_table": mom_table,
            "YoY_table": yoy_table
        }
    
    return all_tables


def drop_interval_column(data):
    for key, tables in data.items():
        for table_name in tables:
            tables[table_name] = tables[table_name].drop(columns=["Interval"])
    return data


# Function to classify based on mapped assets
def classify_key(key, asset_class_map):
    # Split the key into components
    components = key.split("_")
    
    # Iterate over the asset class map
    for asset_class, assets in asset_class_map.items():
        # Check if any component matches the assets in this class
        if any(component in assets for component in components):
            return asset_class  # Return the matching asset class
    
    return "unknown"  # Default if no match is found

def reclassify_titles_cleaned(data, asset_class_map):
    reclassified_data = {}
    for key, value in data.items():
        # Classify the key
        asset_class = classify_key(key, asset_class_map)
        
        # If the asset class is not in the result, initialize it
        if asset_class not in reclassified_data:
            reclassified_data[asset_class] = {"MoM_table": [], "YoY_table": []}
        
        # Append the MoM_table and YoY_table directly to the asset class
        if "MoM_table" in value:
            reclassified_data[asset_class]["MoM_table"].append(value["MoM_table"])
        if "YoY_table" in value:
            reclassified_data[asset_class]["YoY_table"].append(value["YoY_table"])
    
    # Optionally, concatenate tables for each asset class
    for asset_class, tables in reclassified_data.items():
        if tables["MoM_table"]:
            reclassified_data[asset_class]["MoM_table"] = pd.concat(tables["MoM_table"], ignore_index=True)
        else:
            del reclassified_data[asset_class]["MoM_table"]
        if tables["YoY_table"]:
            reclassified_data[asset_class]["YoY_table"] = pd.concat(tables["YoY_table"], ignore_index=True)
        else:
            del reclassified_data[asset_class]["YoY_table"]
    
    return reclassified_data

def group_by_timestamp(data):
    grouped_data = {}
    
    # Iterate over the asset classes (e.g., stocks, cryptocurrency)
    for asset_class, tables in data.items():
        grouped_data[asset_class] = {}
        
        # Process each table (MoM_table and YoY_table)
        for table_name, df in tables.items():
            # Extract timestamp from Portfolio column
            #df['Timestamp'] = df['Portfolio'].str.extract(r'_(\d+[ymax]+)$')[0]
            df["Timestamp"] = df["Portfolio"].str.extract(r"_(\d+[y]|max)$")[0]

            # Group by Timestamp and store in the new structure
            for timestamp, group in df.groupby('Timestamp'):
                if timestamp not in grouped_data[asset_class]:
                    grouped_data[asset_class][timestamp] = {}
                grouped_data[asset_class][timestamp][table_name] = group.drop(columns=['Timestamp'])
    
    return grouped_data

def create_max_correlation_table(data, table_type):
    timeframes = ['2y', '5y', '10y', 'max']
    asset_classes = data.keys()
    
    # Initialize the results table
    result_table = pd.DataFrame(index=timeframes, columns=asset_classes)
    
    # Fill the table
    for asset_class in asset_classes:
        for timeframe in timeframes:
            # Check if the asset class has data for this timeframe and table type
            if timeframe in data[asset_class] and table_type in data[asset_class][timeframe]:
                # Get the table for the timeframe and type
                df = data[asset_class][timeframe][table_type]
                
                # Find the row with the maximum correlation value
                max_row = df.loc[df.iloc[:, 1].idxmax()]  # Assuming correlation is in the 2nd column
                
                # Format: "Ticker: Value"
                result_table.loc[timeframe, asset_class] = f"{max_row['Portfolio']}: {max_row.iloc[1]:.2f}"
    
    return result_table

def create_min_correlation_table(data, table_type):
    timeframes = ['2y', '5y', '10y', 'max']
    asset_classes = data.keys()
    
    # Initialize the results table
    result_table = pd.DataFrame(index=timeframes, columns=asset_classes)
    
    # Fill the table
    for asset_class in asset_classes:
        for timeframe in timeframes:
            # Check if the asset class has data for this timeframe and table type
            if timeframe in data[asset_class] and table_type in data[asset_class][timeframe]:
                # Get the table for the timeframe and type
                df = data[asset_class][timeframe][table_type]
                
                # Find the row with the maximum correlation value
                min_row = df.loc[df.iloc[:, 1].idxmin()]  # Assuming correlation is in the 2nd column
                
                # Format: "Ticker: Value"
                result_table.loc[timeframe, asset_class] = f"{min_row['Portfolio']}: {min_row.iloc[1]:.2f}"
    
    return result_table

def display_table_as_figure(df, title):
    fig, ax = plt.subplots(figsize=(10, len(df) * 0.6))  # Adjust height based on rows
    ax.axis('off')  # Turn off the axis
    ax.axis('tight')  # Tight layout for the table

    # Create the table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=df.index,
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df.columns))))  # Auto-adjust column width

    # Add title
    plt.title(title, fontsize=14, pad=20)
    plt.show()

def display_table_with_colorscale(df, title,save_path=None):
    """
    Displays a DataFrame as a Matplotlib table with a color scale applied to the cells.

    Parameters:
        df (pd.DataFrame): The DataFrame to display.
        title (str): The title for the table.
    """
    # Ensure all numeric values are floats and replace non-numeric with NaN
    def safe_float(x):
        try:
            return float(x.split(": ")[-1])  # Extract numeric value after colon if possible
        except:
            return np.nan  # Replace non-numeric values with NaN

    # Apply numeric conversion to extract numbers where possible
    df_numeric = df.applymap(safe_float)

    # Debugging: Check the converted DataFrame
    print("Converted DataFrame (Numeric):")
    print(df_numeric)

    # Define the color-scaling function
    def cell_color(val):
        if pd.isna(val):  # If value is NaN
            return "white"  # Default background color
        elif val <= 0:
            return "plum"  # Negative values
        elif val > np.median(df_numeric):
            return "skyblue"  # High values
        else:
            return "white"  # Default color for neutral values

    # Create a Matplotlib figure
    fig, ax = plt.subplots(figsize=(15, len(df) * 0.8))  # Adjust figure size dynamically
    ax.axis("off")  # Turn off axes

    # Create the table
    table = ax.table(
        cellText=df.values,  # Original values (including "No valid data")
        colLabels=df.columns,
        rowLabels=df.index,
        cellLoc="center",
        loc="center"
    )

    # Apply colors to cells
    for (row, col), cell in table.get_celld().items():
        if row == 0 or col == -1:  # Skip headers and row labels
            cell.set_fontsize(10)
            cell.set_height(0.4)
            cell.set_text_props(weight="bold")
            continue
        try:
            # Get the numeric value for the cell from df_numeric
            value = df_numeric.iloc[row - 1, col]
            # Apply the color to the cell background
            cell.set_facecolor(cell_color(value))
        except Exception as e:
            print(f"Error applying color to cell ({row}, {col}): {e}")
            cell.set_facecolor("white")  # Default for invalid or non-numeric cells
        
        else: 
            cell.set_height(0.4)

    # Adjust font size and column width
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.auto_set_column_width(col=list(range(len(df.columns))))  # Auto-adjust column width

        
# Add a title
    plt.title(title, fontsize=14, pad=110)

    # Save to the provided PdfPages object, if available
    if pdf is not None:
        pdf.savefig(fig, bbox_inches='tight')
    else:
        plt.show()

    # Close the figure to free up memory
    plt.close(fig)
    
        
    # Display the figure
    plt.show()
    

def display_table_with_tab(df, title):
    """
    Display a DataFrame as a tabular text format using tabulate.
    Converts numeric values, handles non-numeric data, and applies title.

    Parameters:
        df (pd.DataFrame): The DataFrame to display.
        title (str): The title for the table.
    """
    # Ensure all numeric values are floats and replace non-numeric with NaN
    def safe_float(x):
        try:
            return float(x)  # Convert to float if possible
        except:
            return x  # Keep non-numeric values unchanged

    # Convert all values in the DataFrame to numeric where possible
    df_numeric = df.applymap(safe_float)

    # Convert the DataFrame into a tabulated format
    table = tab(
        df_numeric,
        headers="keys",  # Use column headers
        tablefmt="pretty",  # Choose a table format
        showindex=True  # Include row index
    )

    # Print the title and the table
    print("\n" + title + "\n" + "-" * len(title))  # Print the title with a separator
    print(table)


intervals = [monthly]
time_horizon = [two_year, five_year, ten_year, max_year]    

#Download all Data 
test_table = make_data_table(
    smi, nestle, roche, novo_nordisk, lvmh, sap, sp500, apple,  
    nvidia, tsmc, tencent,
    invesco_commodity_composite_ucits_etf, ishares_physical_gold_etf, wisdomtree_brent_crude_oil, 
    ishares_physical_silver_etf, wisdomtree_natural_gas, wisdomtree_wheat, wisdomtree_corn, wisdomtree_soybeans, 
    ishares_global_corporate_bond_ucits_etf, ishares_usd_corporate_bond_ucits_etf, 
    ishares_euro_high_yield_corporate_bond_ucits_etf,
    ishares_euro_inflation_linked_govt_bond_ucits_etf, ubs_etf_us_tips_ucits_etf, ishares_euro_ultrashort_bond_ucits_etf,
    ishares_jp_morgan_em_local_govt_bond_ucits_etf, swiss_prime_site, psp_swiss_property, 
    allreal_holding, mobimo_holding, ubs_etf_sxi_real_estate, 
    procimmo_swiss_commercial_fund, ishares_us_treasury_bond_7_10yr_ucits_etf,
    ishares_us_real_estate_etf, ishares_global_reit_etf, 
    btc, eth, bnb, xrp, ada, 
    interval=intervals, period=time_horizon
    ) 
#test_table = make_data_table(smi, sp500, world_etf, europe_etf, eth, btc, interval=intervals, period=time_horizon)

#Keep all possible portoflios of all asset classes here
all_possible_portfolios_all_asset_classes = []


#Add here which stocks you want to check
#------------STOCKS-------------
stocks = make_asset_class(
    smi, nestle, roche, novo_nordisk, lvmh, sap,
    apple, nvidia, tsmc, tencent
    )

stock_subset = find_subsets(stocks, 2)
all_possible_portfolios_stocks = make_all_portfolios_per_asset_class(stock_subset)

all_possible_portfolios_all_asset_classes.append(all_possible_portfolios_stocks)

#------------COMMODITIES-------------
commodities = make_asset_class(
    invesco_commodity_composite_ucits_etf, ishares_physical_gold_etf, wisdomtree_brent_crude_oil,
    ishares_physical_silver_etf, wisdomtree_natural_gas, wisdomtree_wheat, 
    wisdomtree_corn, wisdomtree_soybeans 
    )

commodities_subset = find_subsets(commodities, 2)
all_possible_portfolios_commodities = make_all_portfolios_per_asset_class(commodities_subset)

all_possible_portfolios_all_asset_classes.append(all_possible_portfolios_commodities)

#------------FIXED INCOME-------------
fixed_income = make_asset_class(
    ishares_global_corporate_bond_ucits_etf, ishares_us_treasury_bond_7_10yr_ucits_etf, 
    ishares_usd_corporate_bond_ucits_etf, ishares_euro_high_yield_corporate_bond_ucits_etf,
    ishares_euro_inflation_linked_govt_bond_ucits_etf, ubs_etf_us_tips_ucits_etf, ishares_euro_ultrashort_bond_ucits_etf,
    ishares_jp_morgan_em_local_govt_bond_ucits_etf
    )

fixed_income_subset = find_subsets(fixed_income, 2)
all_possible_portfolios_fixed_income = make_all_portfolios_per_asset_class(fixed_income_subset)

all_possible_portfolios_all_asset_classes.append(all_possible_portfolios_fixed_income)

#------------REAL ESTATE-------------
real_estate = make_asset_class(
    swiss_prime_site, psp_swiss_property, allreal_holding, mobimo_holding, 
    ubs_etf_sxi_real_estate, procimmo_swiss_commercial_fund, ishares_us_real_estate_etf, ishares_global_reit_etf
    )

real_estate_subset = find_subsets(real_estate, 2)
all_possible_portfolios_real_estate = make_all_portfolios_per_asset_class(real_estate_subset)

all_possible_portfolios_all_asset_classes.append(all_possible_portfolios_real_estate)

#------------CRYPTOCURRENCY------------
crypto = make_asset_class(
    btc, eth, bnb, xrp, ada 
    )

crypto_subset = find_subsets(crypto, 2)
all_possible_portfolios_crypto = make_all_portfolios_per_asset_class(crypto_subset)

all_possible_portfolios_all_asset_classes.append(all_possible_portfolios_crypto)


all_returns = calcualte_beta_for_all(all_possible_portfolios_all_asset_classes)

#CLEAN UP DATA
#drop interval
all_returns_no_intervall = drop_interval_column(all_returns)
#change titles
reclassified_data = reclassify_titles_cleaned(all_returns_no_intervall, asset_class_map)
#group by timestamp
grouped_data = group_by_timestamp(reclassified_data)


# Generate the max MoM and YoY tables
mom_table_max = create_max_correlation_table(grouped_data, 'MoM_table')
yoy_table_max = create_max_correlation_table(grouped_data, 'YoY_table')

# Generate the min MoM and YoY tables
mom_table_min = create_min_correlation_table(grouped_data, 'MoM_table')
yoy_table_min = create_min_correlation_table(grouped_data, 'YoY_table')

# Display the tables as figures
# display_table_as_figure(mom_table_max, "Maximum MoM Correlation Table")
# display_table_as_figure(yoy_table_max, "Maximum YoY Correlation Table")
# display_table_as_figure(mom_table_min, "Minimum MoM Correlation Table")
# display_table_as_figure(yoy_table_min, "Minimum YoY Correlation Table")

# display_table_with_colorscale(mom_table_max, "Maximum MoM Correlation Table")
# display_table_with_colorscale(yoy_table_max, "Maximum YoY Correlation Table")
# display_table_with_colorscale(mom_table_min, "Minimum MoM Correlation Table")
# display_table_with_colorscale(yoy_table_min, "Minimum YoY Correlation Table")


dfs =[
    mom_table_max,
    yoy_table_max,
    mom_table_min,
    yoy_table_min
    
]

titles = ["Maximum MoM Correlation Table" , "Maximum YoY Correlation Table","Minimum MoM Correlation Table", "Minimum YoY Correlation Table" ]

with PdfPages("Results/1_Real_Returns_CorrTables_All_Tickers_.pdf") as pdf:
    for df, title in zip(dfs, titles):
        # Save each table to the PDF
        display_table_with_colorscale(df, title, pdf)
        
        
display_table_with_tab(mom_table_max, "Maximum MoM Correlation Table") # Displays in terminal
