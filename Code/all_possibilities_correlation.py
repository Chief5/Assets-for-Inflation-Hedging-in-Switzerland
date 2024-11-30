import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import os
import seaborn as sns
from sklearn.linear_model import LinearRegression
from itertools import chain, combinations

#STOCK
smi = "^SSMI"
spi_etf = "CHSPI" #does not work
sp500 = "^GSPC"
world_etf= "URTH"
europe_etf= "IEUR"
em_etf ="EEM"

#COMMODITIES
gold = "GLD"
gold_etf = "RING"
gold_ch = "XAUCHF=X"  #does not work

#FIXED INCOME
ten_year_gov_bond = "CH10YT=RR"  #does not work
ch_gov_bond = "AGGS.SW"
ch_corp_bond = "CHBBB.SW" #does not work
tips_bond = "TIP"
treasury_etf = "GOVT"
inflation_bond ="INWG.L"  #does not work
emerg_mark_bond = "EMB"
prime_site = "SPSN.W"  #does not work

#REAL ESTATE
ch_property_etf = "CHREIT.SW"  #does not work
vang_real_est_etf = "VNQ"
dev_eur_prop_etf = "REXP.DE"

#CRYPTOCURRENCY
btc = "BTC-USD"
eth = "ETH-USD"
crypto_index = "BITW"

asset_class_map = {
    "stocks": ["^SSMI", "CHSPI", "^GSPC", "URTH", "IEUR", "EEM"],
    "commodities": ["GLD", "RING", "XAUCHF=X"],
    "fixed_income": ["CH10YT=RR", "AGGS.SW", "CHBBB.SW", "TIP", "GOVT", "INWG.L", "EMB", "SPSN.W"],
    "real_estate": ["CHREIT.SW", "VNQ", "REXP.DE"],
    "cryptocurrency": ["BTC-USD", "ETH-USD", "BITW"]
}

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

def calculate_correlation(portfolio, portfolio_name):
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

    correlation_values = []

    for column in merged_data_mom.columns:
        if column != 'Inflation_Rate_MoM':  # Skip the CPI column itself
            # Calculate correlation with CPI data for the entire dataset
            correlation = merged_data_mom['Inflation_Rate_MoM'].corr(merged_data_mom[column])
            correlation_values.append(correlation)


    for column in merged_data_yoy.columns:
        if column != 'Inflation_Rate_YoY':  # Skip the CPI column itself
            # Calculate correlation with CPI data for the entire dataset
            correlation = merged_data_yoy['Inflation_Rate_YoY'].corr(merged_data_yoy[column])
            correlation_values.append(correlation)
            

    return correlation_values

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
            correlation = calculate_correlation(portfolio_data, portfolio_name)
            
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
        all_portfolios = make_all_portfolios(all, intervalls, time_horizon, test_table)
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

""" def create_max_correlation_table(data, table_type):
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
                result_table.loc[timeframe, asset_class] = f"{max_row['Portfolio']}: {max_row.iloc[1]:.6f}"
    
    return result_table """

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
                
                # Ensure there are valid values
                if df.iloc[:, 1].notna().any():
                    # Find the row with the maximum correlation value
                    max_row = df.loc[df.iloc[:, 1].idxmax()]  # Assuming correlation is in the 2nd column
                    # Format: "Ticker: Value"
                    result_table.loc[timeframe, asset_class] = f"{max_row['Portfolio']}: {max_row.iloc[1]:.6f}"
                else:
                    # Handle case where all values are NaN
                    result_table.loc[timeframe, asset_class] = "No valid data"
    
    return result_table



""" def create_min_correlation_table(data, table_type):
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
                result_table.loc[timeframe, asset_class] = f"{min_row['Portfolio']}: {min_row.iloc[1]:.6f}"
    
    return result_table """

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
                
                # Ensure there are valid values
                if df.iloc[:, 1].notna().any():
                    # Find the row with the minimum correlation value
                    min_row = df.loc[df.iloc[:, 1].idxmin()]  # Assuming correlation is in the 2nd column
                    # Format: "Ticker: Value"
                    result_table.loc[timeframe, asset_class] = f"{min_row['Portfolio']}: {min_row.iloc[1]:.6f}"
                else:
                    # Handle case where all values are NaN
                    result_table.loc[timeframe, asset_class] = "No valid data"
    
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


intervalls = [monthyl]
time_horizon = [two_year, five_year, ten_year, max_year]    

#Download all Data 
test_table = make_data_table(smi, sp500, world_etf, europe_etf, em_etf, gold, gold_etf, ch_gov_bond, tips_bond,treasury_etf, emerg_mark_bond, vang_real_est_etf, btc, eth, interval=intervalls, period=time_horizon)
#test_table = make_data_table(smi, sp500, world_etf, europe_etf, eth, btc, interval=intervalls, period=time_horizon)

#Keep all possible portoflios of all asset classes here
all_possible_portfolios_all_asset_classes = []


#Add here which stocks you want to check
#------------STOCKS-------------
stocks = make_asset_class(smi, sp500, world_etf, europe_etf, em_etf)

stock_subset = find_subsets(stocks, 2)
all_possible_portfolios_stocks = make_all_portfolios_per_asset_class(stock_subset)

all_possible_portfolios_all_asset_classes.append(all_possible_portfolios_stocks)

#------------COMMODITIES-------------
commodities = make_asset_class(gold, gold_etf)

commodities_subset = find_subsets(commodities, 2)
all_possible_portfolios_commodities = make_all_portfolios_per_asset_class(commodities_subset)

all_possible_portfolios_all_asset_classes.append(all_possible_portfolios_commodities)

#------------FIXED INCOME-------------
fixed_income = make_asset_class(ch_gov_bond, tips_bond, treasury_etf, emerg_mark_bond)

fixed_income_subset = find_subsets(fixed_income, 2)
all_possible_portfolios_fixed_income = make_all_portfolios_per_asset_class(fixed_income_subset)

all_possible_portfolios_all_asset_classes.append(all_possible_portfolios_fixed_income)

#------------REAL ESTATE-------------
real_estate = make_asset_class(vang_real_est_etf)

real_estate_subset = find_subsets(real_estate, 1)
all_possible_portfolios_real_estate = make_all_portfolios_per_asset_class(real_estate_subset)

all_possible_portfolios_all_asset_classes.append(all_possible_portfolios_real_estate)

#------------CRYPTOCURRENCY------------
crypto = make_asset_class(btc, eth)

crypto_subset = find_subsets(crypto, 2)
all_possible_portfolios_crypto = make_all_portfolios_per_asset_class(crypto_subset)

all_possible_portfolios_all_asset_classes.append(all_possible_portfolios_crypto)


all_returns = calcualte_beta_for_all(all_possible_portfolios_all_asset_classes)

#CLEANU UP DATA
#drop interval
all_returns_no_intervall = drop_interval_column(all_returns)
#change titles
reclassified_data = reclassify_titles_cleaned(all_returns_no_intervall, asset_class_map)
print(reclassified_data)
#group by timestamp
grouped_data = group_by_timestamp(reclassified_data)


# Generate the max MoM and YoY tables
mom_table_max = create_max_correlation_table(grouped_data, 'MoM_table')
yoy_table_max = create_max_correlation_table(grouped_data, 'YoY_table')

# Generate the min MoM and YoY tables
mom_table_min = create_min_correlation_table(grouped_data, 'MoM_table')
yoy_table_min = create_min_correlation_table(grouped_data, 'YoY_table')

# Display the tables as figures
display_table_as_figure(mom_table_max, "Maximum MoM Correlation Table")
display_table_as_figure(yoy_table_max, "Maximum YoY Correlation Table")
display_table_as_figure(mom_table_min, "Minimum MoM Correlation Table")
display_table_as_figure(yoy_table_min, "Minimum YoY Correlation Table")

