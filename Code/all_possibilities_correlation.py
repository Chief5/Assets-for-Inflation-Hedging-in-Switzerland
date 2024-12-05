import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import os
import seaborn as sns
from sklearn.linear_model import LinearRegression
from itertools import chain, combinations

#STOCKS
#SMI
smi = "^SSMI"
abb = "ABBN.SW"
alcon = "ALC.SW"
richemont = "CFR.SW"
credit_suisse = "CSGN.SW"
geberit = "GEBN.SW"
givaudan = "GIVN.SW"
holcim = "HOLN.SW"
logitech = "LOGN.SW"
lonza = "LONN.SW"
nestle = "NESN.SW"
novartis = "NOVN.SW"
partners_group = "PGHN.SW"
roche = "ROG.SW"
sika = "SIKA.SW"
sonova = "SOON.SW"
swisscom = "SCMN.SW"
swiss_life = "SLHN.SW"
swiss_re = "SREN.SW"
ubs = "UBSG.SW"
zurich_insurance = "ZURN.SW"
# Top 20 European Companies by Market Capitalization
novo_nordisk = "NVO"
lvmh = "MC.PA"
sap = "SAP"
asml = "ASML"
hermes = "RMS.PA"
loreal = "OR.PA"
totalenergies = "TTE"
astrazeneca = "AZN"
shell = "SHEL"
linde = "LIN"
siemens = "SIE.DE"
unilever = "UL"
airbus = "AIR.PA"
schneider_electric = "SU.PA"
banco_santander = "SAN"
enel = "ENEL.MI"
bp = "BP"
air_liquide = "AI.PA"
allianz = "ALV.DE"
diageo = "DEO"
#S&P500 (Top 20 S&P 500 Constituents)
sp500 = "^GSPC"
apple = "AAPL"
microsoft = "MSFT"
amazon = "AMZN"
alphabet_a = "GOOGL"
alphabet_c = "GOOG"
berkshire_hathaway = "BRK.B"
meta_platforms = "META"
tesla = "TSLA"
nvidia = "NVDA"
jpmorgan_chase = "JPM"
johnson_and_johnson = "JNJ"
visa = "V"
procter_and_gamble = "PG"
unitedhealth_group = "UNH"
home_depot = "HD"
mastercard = "MA"
exxon_mobil = "XOM"
chevron = "CVX"
pfizer = "PFE"
coca_cola = "KO"
# Top 20 Asian Companies by Market Capitalization
tsmc = "TSM"
tencent = "TCEHY"
alibaba = "BABA"
reliance = "RELIANCE.NS"
samsung = "005930.KS"
meituan = "3690.HK"
kweichow_moutai = "600519.SS"
icici_bank = "ICICIBANK.NS"
hdfc_bank = "HDFCBANK.NS"
china_construction_bank = "0939.HK"
ping_an = "2318.HK"
sony = "6758.T"
toyota = "7203.T"
icbc = "1398.HK"
agricultural_bank_of_china = "1288.HK"
bank_of_china = "3988.HK"
petrochina = "0857.HK"
china_mobile = "0941.HK"
hon_hai = "2317.TW"
jd = "9618.HK"

#COMMODITIES
# Broad Commodity ETFs
ishares_bloomberg_commodity_ucits_etf = "ICOM.SW"
wisdomtree_bloomberg_commodity_ucits_etf = "WCOA.SW"
ubs_cmci_composite_sf_ucits_etf = "CCPHA.SW"
invesco_commodity_composite_ucits_etf = "COMC.SW"
lyxor_commodities_crb_tr_ucits_etf = "CRB.SW"
# Precious Metals ETFs
ishares_physical_gold_etf = "IGLN.SW"
wisdomtree_physical_gold = "PHAU.SW"
ubs_etf_gold = "AUUS.SW"
invesco_physical_gold_etc = "SGLD.SW"
xetra_gold = "4GLD.SW"
# Energy ETFs
ishares_oil_gas_exploration_production_ucits_etf = "SPOG.SW"
wisdomtree_brent_crude_oil = "BRNT.SW"
ubs_etf_cmci_energy_sf_ucits_etf = "CENUS.SW"
invesco_morningstar_energy_sector_ucits_etf = "XENS.SW"
lyxor_commodities_crude_oil_ucits_etf = "CRU.SW"
# Agriculture ETFs
ishares_agribusiness_ucits_etf = "SAGR.SW"
wisdomtree_commodities_sectors_agribusiness = "AGRI.SW"
ubs_etf_cmci_agriculture_sf_ucits_etf = "CAGUS.SW"
invesco_morningstar_agribusiness_ucits_etf = "FARM.SW"
lyxor_commodities_rogers_agriculture_ucits_etf = "AGRO.SW"
# Industrial Metals ETFs
ishares_physical_silver_etf = "ISLN.SW"
wisdomtree_physical_silver = "PHAG.SW"
ubs_etf_cmci_industrial_metals_sf_ucits_etf = "CIMUS.SW"
invesco_physical_silver_etc = "SSIL.SW"
xetra_silver = "4SIL.SW"
# Specific Commodity ETFs
wisdomtree_copper = "COPA.SW"
wisdomtree_natural_gas = "NGAS.SW"
wisdomtree_wheat = "WEAT.SW"
wisdomtree_corn = "CORN.SW"
wisdomtree_soybeans = "SOYB.SW"
# Leveraged and Inverse Commodity ETFs
wisdomtree_wti_crude_oil_2x_daily_leveraged = "LOIL.SW"
wisdomtree_wti_crude_oil_1x_daily_short = "SOIL.SW"
wisdomtree_natural_gas_2x_daily_leveraged = "LNGA.SW"
wisdomtree_natural_gas_1x_daily_short = "SNGA.SW"
wisdomtree_gold_2x_daily_leveraged = "LBUL.SW"
# Commodity Equity ETFs
ishares_global_commodities_equity_ucits_etf = "COAU.SW"
lyxor_commodities_crb_non_energies_ucits_etf = "CRBN.SW"
invesco_morningstar_global_commodities_ucits_etf = "GCCM.SW"
ubs_etf_cmci_ex_energy_sf_ucits_etf = "CEXUS.SW"
wisdomtree_commodity_securities_limited = "COMS.SW"
# Commodity Futures ETFs
wisdomtree_commodity_futures = "COMF.SW"
lyxor_commodities_rogers_international_ucits_etf = "ROGI.SW"
invesco_commodity_futures_ucits_etf = "FUTR.SW"
ubs_etf_cmci_commodity_futures_sf_ucits_etf = "CCFUS.SW"
ishares_commodity_futures_strategy_ucits_etf = "ICOMF.SW"
# Commodity Currency-Hedged ETFs
ishares_bloomberg_commodity_ucits_etf_eur_hedged = "ICOMH.SW"
wisdomtree_bloomberg_commodity_ucits_etf_gbp_hedged = "WCOG.SW"
ubs_etf_cmci_composite_sf_ucits_etf_chf_hedged = "CCPHC.SW"
invesco_commodity_composite_ucits_etf_usd_hedged = "COMCU.SW"
lyxor_commodities_crb_tr_ucits_etf_jpy_hedged = "CRBJ.SW"


#FIXED INCOME SECURITIES
# Broad Market Bond ETFs
ishares_core_global_aggregate_bond_ucits_etf = "AGGG.SW"
vanguard_global_bond_index_fund_ucits_etf = "VAGB.SW"
ishares_global_corporate_bond_ucits_etf = "CORP.SW"
spdr_bloomberg_barclays_global_aggregate_bond_ucits_etf = "GLAG.SW"
ubs_etf_global_aggregate_bond_ucits_etf = "AGGUS.SW"
# Government Bond ETFs
ishares_switzerland_govt_bond_ucits_etf = "CHGOV.SW"
ubs_etf_swiss_govt_bond_1_3_ucits_etf = "CH13.SW"
spdr_bloomberg_barclays_1_3_year_us_treasury_ucits_etf = "TSY3.SW"
ishares_us_treasury_bond_7_10yr_ucits_etf = "IBTM.SW"
vanguard_us_government_bond_index_ucits_etf = "VUTY.SW"
# Corporate Bond ETFs
ishares_euro_corporate_bond_large_cap_ucits_etf = "IBCX.SW"
spdr_bloomberg_barclays_euro_corporate_bond_ucits_etf = "SYBC.SW"
vanguard_euro_investment_grade_bond_index_ucits_etf = "VEIG.SW"
ubs_etf_cmci_corporate_bond_ucits_etf = "CBND.SW"
ishares_usd_corporate_bond_ucits_etf = "LQDE.SW"
# High Yield Bond ETFs
ishares_euro_high_yield_corporate_bond_ucits_etf = "IHYG.SW"
spdr_bloomberg_barclays_euro_high_yield_bond_ucits_etf = "SYBJ.SW"
vanguard_usd_emerging_markets_government_bond_ucits_etf = "VEMT.SW"
ubs_etf_jp_morgan_usd_emerging_markets_bond_ucits_etf = "EMB.SW"
ishares_usd_emerging_markets_bond_ucits_etf = "IEMB.SW"
# Inflation-Linked Bond ETFs
ishares_euro_inflation_linked_govt_bond_ucits_etf = "IILG.SW"
spdr_bloomberg_barclays_euro_inflation_linked_bond_ucits_etf = "SYBI.SW"
vanguard_us_tips_ucits_etf = "VTIP.SW"
ubs_etf_us_tips_ucits_etf = "TIPS.SW"
ishares_us_tips_ucits_etf = "ITPS.SW"
# Short Duration Bond ETFs
ishares_euro_ultrashort_bond_ucits_etf = "ERNE.SW"
spdr_bloomberg_barclays_0_3_year_us_corporate_bond_ucits_etf = "SYBU.SW"
vanguard_usd_corporate_1_3_year_bond_ucits_etf = "VUSC.SW"
ubs_etf_usd_corporate_1_3_year_bond_ucits_etf = "UC13.SW"
ishares_usd_short_duration_corporate_bond_ucits_etf = "SDIG.SW"
# Emerging Markets Bond ETFs
ishares_jp_morgan_em_local_govt_bond_ucits_etf = "IEML.SW"
spdr_bloomberg_barclays_emerging_markets_local_bond_ucits_etf = "SYBE.SW"
vanguard_emerging_markets_bond_ucits_etf = "VEMB.SW"
ubs_etf_jp_morgan_emerging_markets_bond_ucits_etf = "EMUS.SW"
ishares_emerging_markets_local_govt_bond_ucits_etf = "EMLG.SW"
# Corporate Bond ETFs by Maturity
invesco_bulletshares_2024_corporate_bond_ucits_etf = "BSCO.SW"
invesco_bulletshares_2025_corporate_bond_ucits_etf = "BSCP.SW"
invesco_bulletshares_2026_corporate_bond_ucits_etf = "BSCQ.SW"
invesco_bulletshares_2027_corporate_bond_ucits_etf = "BSCR.SW"
invesco_bulletshares_2028_corporate_bond_ucits_etf = "BSCS.SW"
# High Yield Bond ETFs by Maturity
invesco_bulletshares_2024_high_yield_corporate_bond_ucits_etf = "BSJO.SW"
invesco_bulletshares_2025_high_yield_corporate_bond_ucits_etf = "BSJP.SW"
invesco_bulletshares_2026_high_yield_corporate_bond_ucits_etf = "BSJQ.SW"
invesco_bulletshares_2027_high_yield_corporate_bond_ucits_etf = "BSJR.SW"
invesco_bulletshares_2028_high_yield_corporate_bond_ucits_etf = "BSJS.SW"
# Aggregate Bond ETFs
ishares_global_aggregate_bond_ucits_etf = "AGGG.SW"
vanguard_global_aggregate_bond_ucits_etf = "VAGG.SW"
spdr_bloomberg_barclays_global_aggregate_bond_ucits_etf = "GLAG.SW"
ubs_etf_global_aggregate_bond_ucits_etf = "AGGUS.SW"
ishares_usd_aggregate_bond_ucits_etf = "AGGU.SW"

#REAL ESTATE
# Swiss Real Estate Companies
swiss_prime_site = "SPSN.SW"
psp_swiss_property = "PSPN.SW"
allreal_holding = "ALLN.SW"
mobimo_holding = "MOBN.SW"
zug_estates_holding = "ZUGN.SW"
investis_holding = "IREN.SW"
intershop_holding = "ISN.SW"
hiag_immobilien = "HIAG.SW"
plazza_ag = "PLAN.SW"
sf_urban_properties = "SFPN.SW"
warteck_invest = "WARN.SW"
fundamenta_real_estate = "FREN.SW"
varia_us_properties = "VARN.SW"
novavest_real_estate = "NREN.SW"
zueblin_immobilien = "ZUBN.SW"
ina_invest_holding = "INA.SW"
# Swiss Real Estate Funds
ubs_etf_sxi_real_estate = "SRECHA.SW"
credit_suisse_real_estate_fund_siat = "CSREF.SW"
ubs_property_fund_residential = "UBSPF.SW"
credit_suisse_real_estate_fund_livingplus = "CSLP.SW"
credit_suisse_real_estate_fund_green_property = "CSGP.SW"
ubs_property_fund_mixed_sima = "UBSMS.SW"
swisscanto_real_estate_fund = "SWCRF.SW"
ubs_ast_immobilien_schweiz = "UBSIS.SW"
la_fonciere = "LFON.SW"
procimmo_swiss_commercial_fund = "PSCF.SW"
# International Real Estate ETFs
vanguard_real_estate_etf = "VNQ"
ishares_us_real_estate_etf = "IYR"
spdr_dow_jones_reit_etf = "RWR"
schwab_us_reit_etf = "SCHH"
ishares_global_reit_etf = "REET"
vanguard_global_ex_us_real_estate_etf = "VNQI"
spdr_dow_jones_global_real_estate_etf = "RWO"
spdr_dow_jones_international_real_estate_etf = "RWX"
land_securities_group = "LAND.L"
unibail_rodamco_westfield = "URW.AS"
vonovia_se = "VNA.DE"
leg_immobilien = "LEG.DE"
deutsche_wohnen = "DWNI.DE"
grand_city_properties = "GYC.DE"


#CRYPTOCURRENCY
btc = "BTC-USD"
eth = "ETH-USD"
bnb = "BNB-USD"
xrp = "XRP-USD"
ada = "ADA-USD"
doge = "DOGE-USD"
sol = "SOL-USD"
dot = "DOT-USD"
matic = "MATIC-USD"
ltc = "LTC-USD"

asset_class_map = {
    "stocks": ["^SSMI", "ABBN.SW", "ALC.SW", "CFR.SW", "CSGN.SW", "GEBN.SW", "GIVN.SW", "HOLN.SW", 
    "LOGN.SW", "LONN.SW", "NESN.SW", "NOVN.SW", "PGHN.SW", "ROG.SW", "SIKA.SW", "SOON.SW", 
    "SCMN.SW", "SLHN.SW", "SREN.SW", "UBSG.SW", "ZURN.SW", "NVO", "MC.PA", "SAP", "ASML", 
    "RMS.PA", "OR.PA", "TTE", "AZN", "SHEL", "LIN", "SIE.DE", "UL", "AIR.PA", "SU.PA", 
    "SAN", "ENEL.MI", "BP", "AI.PA", "ALV.DE", "DEO", "AAPL", "MSFT", "AMZN", "GOOGL", 
    "GOOG", "BRK.B", "META", "TSLA", "NVDA", "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", 
    "XOM", "CVX", "PFE", "KO", "TSM", "TCEHY", "BABA", "RELIANCE.NS", "005930.KS", "3690.HK", 
    "600519.SS", "ICICIBANK.NS", "HDFCBANK.NS", "0939.HK", "2318.HK", "6758.T", "7203.T", 
    "1398.HK", "1288.HK", "3988.HK", "0857.HK", "0941.HK", "2317.TW", "9618.HK"],
    "commodities": ["ICOM.SW", "WCOA.SW", "CCPHA.SW", "COMC.SW", "CRB.SW",
    "IGLN.SW", "PHAU.SW", "AUUS.SW", "SGLD.SW", "4GLD.SW",
    "SPOG.SW", "BRNT.SW", "CENUS.SW", "XENS.SW", "CRU.SW",
    "SAGR.SW", "AGRI.SW", "CAGUS.SW", "FARM.SW", "AGRO.SW",
    "ISLN.SW", "PHAG.SW", "CIMUS.SW", "SSIL.SW", "4SIL.SW",
    "COPA.SW", "NGAS.SW", "WEAT.SW", "CORN.SW", "SOYB.SW",
    "LOIL.SW", "SOIL.SW", "LNGA.SW", "SNGA.SW", "LBUL.SW",
    "COAU.SW", "CRBN.SW", "GCCM.SW", "CEXUS.SW", "COMS.SW",
    "COMF.SW", "ROGI.SW", "FUTR.SW", "CCFUS.SW", "ICOMF.SW",
    "ICOMH.SW", "WCOG.SW", "CCPHC.SW", "COMCU.SW", "CRBJ.SW"],
    "fixed_income": ["AGGG.SW", "VAGB.SW", "CORP.SW", "GLAG.SW", "AGGUS.SW",
    "CHGOV.SW", "CH13.SW", "TSY3.SW", "IBTM.SW", "VUTY.SW",
    "IBCX.SW", "SYBC.SW", "VEIG.SW", "CBND.SW", "LQDE.SW",
    "IHYG.SW", "SYBJ.SW", "VEMT.SW", "EMB.SW", "IEMB.SW",
    "IILG.SW", "SYBI.SW", "VTIP.SW", "TIPS.SW", "ITPS.SW",
    "ERNE.SW", "SYBU.SW", "VUSC.SW", "UC13.SW", "SDIG.SW",
    "IEML.SW", "SYBE.SW", "VEMB.SW", "EMUS.SW", "EMLG.SW",
    "BSCO.SW", "BSCP.SW", "BSCQ.SW", "BSCR.SW", "BSCS.SW",
    "BSJO.SW", "BSJP.SW", "BSJQ.SW", "BSJR.SW", "BSJS.SW",
    "AGGG.SW", "VAGG.SW", "GLAG.SW", "AGGUS.SW", "AGGU.SW"],
    "real_estate": ["SPSN.SW", "PSPN.SW", "ALLN.SW", "MOBN.SW", 
    "ZUGN.SW", "IREN.SW", "ISN.SW", "HIAG.SW", 
    "PLAN.SW", "SFPN.SW", "WARN.SW", "FREN.SW", 
    "VARN.SW", "NREN.SW", "ZUBN.SW", "INA.SW",  
    "SRECHA.SW", "CSREF.SW", "UBSPF.SW", "CSLP.SW", 
    "CSGP.SW", "UBSMS.SW", "SWCRF.SW", "UBSIS.SW", 
    "LFON.SW", "PSCF.SW", "VNQ", "IYR", 
    "RWR", "SCHH", "REET", "VNQI", 
    "RWO", "RWX", "LAND.L", "URW.AS", 
    "VNA.DE", "LEG.DE", "DWNI.DE", "GYC.DE"],
    "cryptocurrency": ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", 
    "DOGE-USD", "SOL-USD", "DOT-USD", "MATIC-USD", "LTC-USD"]
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
test_table = make_data_table(smi, sp500, btc, eth, interval=intervalls, period=time_horizon)
#test_table = make_data_table(smi, sp500, world_etf, europe_etf, eth, btc, interval=intervalls, period=time_horizon)

#Keep all possible portoflios of all asset classes here
all_possible_portfolios_all_asset_classes = []


#Add here which stocks you want to check
#------------STOCKS-------------
stocks = make_asset_class(
    smi, abb, alcon, richemont, credit_suisse, geberit, givaudan, holcim, logitech,
    lonza, nestle, novartis, partners_group, roche, sika, sonova, swisscom, swiss_life,
    swiss_re, ubs, zurich_insurance, novo_nordisk, lvmh, sap, asml, hermes, loreal,
    totalenergies, astrazeneca, shell, linde, siemens, unilever, airbus, schneider_electric,
    banco_santander, enel, bp, air_liquide, allianz, diageo, apple, microsoft, amazon,
    alphabet_a, alphabet_c, berkshire_hathaway, meta_platforms, tesla, nvidia, jpmorgan_chase,
    johnson_and_johnson, visa, procter_and_gamble, unitedhealth_group, home_depot, mastercard,
    exxon_mobil, chevron, pfizer, coca_cola, tsmc, tencent, alibaba, reliance, samsung,
    meituan, kweichow_moutai, icici_bank, hdfc_bank, china_construction_bank, ping_an, sony,
    toyota, icbc, agricultural_bank_of_china, bank_of_china, petrochina, china_mobile, hon_hai, jd)

stock_subset = find_subsets(stocks, 1)
all_possible_portfolios_stocks = make_all_portfolios_per_asset_class(stock_subset)

all_possible_portfolios_all_asset_classes.append(all_possible_portfolios_stocks)

#------------COMMODITIES-------------
commodities = make_asset_class(
    ishares_bloomberg_commodity_ucits_etf, wisdomtree_bloomberg_commodity_ucits_etf, ubs_cmci_composite_sf_ucits_etf, 
    invesco_commodity_composite_ucits_etf, lyxor_commodities_crb_tr_ucits_etf, 
    ishares_physical_gold_etf, wisdomtree_physical_gold, ubs_etf_gold, invesco_physical_gold_etc, xetra_gold, 
    ishares_oil_gas_exploration_production_ucits_etf, wisdomtree_brent_crude_oil, ubs_etf_cmci_energy_sf_ucits_etf, 
    invesco_morningstar_energy_sector_ucits_etf, lyxor_commodities_crude_oil_ucits_etf, 
    ishares_agribusiness_ucits_etf, wisdomtree_commodities_sectors_agribusiness, ubs_etf_cmci_agriculture_sf_ucits_etf, 
    invesco_morningstar_agribusiness_ucits_etf, lyxor_commodities_rogers_agriculture_ucits_etf, 
    ishares_physical_silver_etf, wisdomtree_physical_silver, ubs_etf_cmci_industrial_metals_sf_ucits_etf, 
    invesco_physical_silver_etc, xetra_silver, 
    wisdomtree_copper, wisdomtree_natural_gas, wisdomtree_wheat, wisdomtree_corn, wisdomtree_soybeans, 
    wisdomtree_wti_crude_oil_2x_daily_leveraged, wisdomtree_wti_crude_oil_1x_daily_short, wisdomtree_natural_gas_2x_daily_leveraged, 
    wisdomtree_natural_gas_1x_daily_short, wisdomtree_gold_2x_daily_leveraged, 
    ishares_global_commodities_equity_ucits_etf, lyxor_commodities_crb_non_energies_ucits_etf, invesco_morningstar_global_commodities_ucits_etf, 
    ubs_etf_cmci_ex_energy_sf_ucits_etf, wisdomtree_commodity_securities_limited, 
    wisdomtree_commodity_futures, lyxor_commodities_rogers_international_ucits_etf, invesco_commodity_futures_ucits_etf, 
    ubs_etf_cmci_commodity_futures_sf_ucits_etf, ishares_commodity_futures_strategy_ucits_etf, 
    ishares_bloomberg_commodity_ucits_etf_eur_hedged, wisdomtree_bloomberg_commodity_ucits_etf_gbp_hedged, 
    ubs_etf_cmci_composite_sf_ucits_etf_chf_hedged, invesco_commodity_composite_ucits_etf_usd_hedged, 
    lyxor_commodities_crb_tr_ucits_etf_jpy_hedged)

commodities_subset = find_subsets(commodities, 1)
all_possible_portfolios_commodities = make_all_portfolios_per_asset_class(commodities_subset)

all_possible_portfolios_all_asset_classes.append(all_possible_portfolios_commodities)

#------------FIXED INCOME-------------
fixed_income = make_asset_class(ishares_core_global_aggregate_bond_ucits_etf, vanguard_global_bond_index_fund_ucits_etf, 
    ishares_global_corporate_bond_ucits_etf, spdr_bloomberg_barclays_global_aggregate_bond_ucits_etf, 
    ubs_etf_global_aggregate_bond_ucits_etf, ishares_switzerland_govt_bond_ucits_etf, 
    ubs_etf_swiss_govt_bond_1_3_ucits_etf, spdr_bloomberg_barclays_1_3_year_us_treasury_ucits_etf, 
    ishares_us_treasury_bond_7_10yr_ucits_etf, vanguard_us_government_bond_index_ucits_etf, 
    ishares_euro_corporate_bond_large_cap_ucits_etf, spdr_bloomberg_barclays_euro_corporate_bond_ucits_etf, 
    vanguard_euro_investment_grade_bond_index_ucits_etf, ubs_etf_cmci_corporate_bond_ucits_etf, 
    ishares_usd_corporate_bond_ucits_etf, ishares_euro_high_yield_corporate_bond_ucits_etf, 
    spdr_bloomberg_barclays_euro_high_yield_bond_ucits_etf, vanguard_usd_emerging_markets_government_bond_ucits_etf, 
    ubs_etf_jp_morgan_usd_emerging_markets_bond_ucits_etf, ishares_usd_emerging_markets_bond_ucits_etf, 
    ishares_euro_inflation_linked_govt_bond_ucits_etf, spdr_bloomberg_barclays_euro_inflation_linked_bond_ucits_etf, 
    vanguard_us_tips_ucits_etf, ubs_etf_us_tips_ucits_etf, ishares_us_tips_ucits_etf, 
    ishares_euro_ultrashort_bond_ucits_etf, spdr_bloomberg_barclays_0_3_year_us_corporate_bond_ucits_etf, 
    vanguard_usd_corporate_1_3_year_bond_ucits_etf, ubs_etf_usd_corporate_1_3_year_bond_ucits_etf, 
    ishares_usd_short_duration_corporate_bond_ucits_etf, ishares_jp_morgan_em_local_govt_bond_ucits_etf, 
    spdr_bloomberg_barclays_emerging_markets_local_bond_ucits_etf, vanguard_emerging_markets_bond_ucits_etf, 
    ubs_etf_jp_morgan_emerging_markets_bond_ucits_etf, ishares_emerging_markets_local_govt_bond_ucits_etf, 
    invesco_bulletshares_2024_corporate_bond_ucits_etf, invesco_bulletshares_2025_corporate_bond_ucits_etf, 
    invesco_bulletshares_2026_corporate_bond_ucits_etf, invesco_bulletshares_2027_corporate_bond_ucits_etf, 
    invesco_bulletshares_2028_corporate_bond_ucits_etf, invesco_bulletshares_2024_high_yield_corporate_bond_ucits_etf, 
    invesco_bulletshares_2025_high_yield_corporate_bond_ucits_etf, invesco_bulletshares_2026_high_yield_corporate_bond_ucits_etf, 
    invesco_bulletshares_2027_high_yield_corporate_bond_ucits_etf, invesco_bulletshares_2028_high_yield_corporate_bond_ucits_etf, 
    ishares_global_aggregate_bond_ucits_etf, vanguard_global_aggregate_bond_ucits_etf, 
    spdr_bloomberg_barclays_global_aggregate_bond_ucits_etf, ubs_etf_global_aggregate_bond_ucits_etf, 
    ishares_usd_aggregate_bond_ucits_etf)

fixed_income_subset = find_subsets(fixed_income, 1)
all_possible_portfolios_fixed_income = make_all_portfolios_per_asset_class(fixed_income_subset)

all_possible_portfolios_all_asset_classes.append(all_possible_portfolios_fixed_income)

#------------REAL ESTATE-------------
real_estate = make_asset_class(swiss_prime_site, psp_swiss_property, allreal_holding, mobimo_holding, zug_estates_holding, investis_holding, 
    intershop_holding, hiag_immobilien, plazza_ag, sf_urban_properties, warteck_invest, fundamenta_real_estate, 
    varia_us_properties, novavest_real_estate, zueblin_immobilien, ina_invest_holding, 
    ubs_etf_sxi_real_estate, credit_suisse_real_estate_fund_siat, ubs_property_fund_residential, 
    credit_suisse_real_estate_fund_livingplus, credit_suisse_real_estate_fund_green_property, 
    ubs_property_fund_mixed_sima, swisscanto_real_estate_fund, ubs_ast_immobilien_schweiz, 
    la_fonciere, procimmo_swiss_commercial_fund, vanguard_real_estate_etf, 
    ishares_us_real_estate_etf, spdr_dow_jones_reit_etf, schwab_us_reit_etf, ishares_global_reit_etf, 
    vanguard_global_ex_us_real_estate_etf, spdr_dow_jones_global_real_estate_etf, 
    spdr_dow_jones_international_real_estate_etf, land_securities_group, unibail_rodamco_westfield, 
    vonovia_se, leg_immobilien, deutsche_wohnen, grand_city_properties)

real_estate_subset = find_subsets(real_estate, 1)
all_possible_portfolios_real_estate = make_all_portfolios_per_asset_class(real_estate_subset)

all_possible_portfolios_all_asset_classes.append(all_possible_portfolios_real_estate)

#------------CRYPTOCURRENCY------------
crypto = make_asset_class(btc, eth, bnb, xrp, ada, doge, sol, dot, matic, ltc)

crypto_subset = find_subsets(crypto, 1)
all_possible_portfolios_crypto = make_all_portfolios_per_asset_class(crypto_subset)

all_possible_portfolios_all_asset_classes.append(all_possible_portfolios_crypto)


all_returns = calcualte_beta_for_all(all_possible_portfolios_all_asset_classes)

#CLEAN UP DATA
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

