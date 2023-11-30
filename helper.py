import numpy as np
import pandas as pd

# Backtest nav curve generator
def cal_nav(df_port, close, fee=0):

    # -------------------------------------------------------------------------
    # Backtest period and related dates
    # -------------------------------------------------------------------------
    # Global daily frequency series
    daily_dates = close.index.tolist()
    
    # Panel dates (i.e. dates when signals are given)
    panel_dates = df_port.index.tolist()
    
    # Refresh dates
    refresh_dates = [] # Store refresh dates, i.e. the next trading day closest to the signal date
    check_pos = 0 # Mark the search point of the global sequence in order to find the position of the signal date in the global sequence
    for date in panel_dates:
        if check_pos >= len(daily_dates): break # The entire global sequence has been searched
        try:
            while date >= daily_dates[check_pos]:
                check_pos += 1
            else:
                refresh_dates.append(daily_dates[check_pos])
        except: # The last panel may report an error
            break
            
    # Generate mapping matrix between refresh dates and panel dates
    refresh_dict = dict(zip(refresh_dates,panel_dates))
    
    # The first trading day of the backtest period is the first refresh date
    start_date = refresh_dates[0]
    
    # The end of the backtest period is the last day of the daily frequency sequence
    end_date = daily_dates[-1]
    
    # Get the closing price sequence of the target interval
    backtest_close = close.loc[start_date:end_date,:]
    
    # Backtest dates
    backtest_dates = backtest_close.index.tolist()
    
    # -------------------------------------------------------------------------
    # Update the nav curve
    # -------------------------------------------------------------------------
    # Initialize the nav sequence
    nav = pd.Series(index=[panel_dates[0]]+backtest_dates, name='nav', dtype=float)
    nav.iloc[0] = 1.0 # The starting point of the nav sequence is 1
    
    # Initialize bilateral turnover
    turnover_tot = 0.0
    
    # Traverse each backtest date and generate the latest nav
    for date_index in range(len(backtest_dates)):
        
        # Specific date
        date = backtest_dates[date_index]
        
        # If it is the first day of the backtest, that is, the first refresh date, then you need to build the initial position
        if date_index == 0:
            
            panel_date = refresh_dict[date]
            
            new_weight = df_port.loc[panel_date, :]
            
            portfolio = (1 - fee) * new_weight
            
            nav[date] = np.nansum(portfolio)
            
            continue
    
        # If it is not the first day of the backtest, then you need to update the nav curve
        cur_close = backtest_close.iloc[date_index,:]
        
        prev_close = backtest_close.iloc[date_index-1,:]
        
        portfolio = cur_close / prev_close * portfolio
        
        nav[date] = np.nansum(portfolio)
        
        # If the current date is a refresh date, then you need to adjust the position
        if date in refresh_dates:
            
            # Get the old normalized weight
            old_weight = portfolio / np.nansum(portfolio)
            old_weight[old_weight.isnull()] = 0
            
            # Get the new normalized weight
            new_weight = df_port.loc[refresh_dict[date], :]
    
            # Calculate the turnover rate
            turn_over = np.sum(np.abs(new_weight-old_weight))
            turnover_tot += turn_over * 0.5
            
            # Update the nav curve
            nav[date] = nav[date] * (1 - turn_over * fee)
            
            # Build a new position
            portfolio = new_weight * nav[date]
    
    return nav, turnover_tot / len(refresh_dates)

# Annualized Return
def annualized_return(nav):
    
    return pow(nav.iloc[-1] / nav.iloc[0], 250/len(nav)) - 1

# Annualized Volatility
def annualized_vol(nav):
    
    return nav.pct_change().dropna().std() * np.sqrt(250)

# Annualized Sharpe Ratio
def sharp_ratio(nav):
    
    return annualized_return(nav) / annualized_vol(nav)

# Max Drawdown
def max_drawndown(nav):
    
    drawdown = 0
    
    for index in range(1,len(nav)):
        
        cur_drawndown = nav.iloc[index] / max(nav.iloc[0:index]) - 1
        
        if cur_drawndown < drawdown:
            
            drawdown = cur_drawndown
        
    return drawdown

# Calmar Ratio
def calmar_ratio(nav):
    
    return annualized_return(nav) / -max_drawndown(nav)

# backtest performance
def performance(nav):

    perf = pd.Series({'Annualized Return' : annualized_return(nav),
                      'Annualized volatility' : annualized_vol(nav),
                      'Sharpe ratio'   : sharp_ratio(nav),
                      'Max Drawdown'   : max_drawndown(nav),
                      'Calmar Ratio'   : calmar_ratio(nav)},
                    name = 'Performance Indicators')
    return perf