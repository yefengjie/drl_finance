import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import syntheticChrissAlmgren as sca

from statsmodels.iolib.table import SimpleTable
from statsmodels.compat.python import zip_longest
from statsmodels.iolib.tableformatting import fmt_2cols


def generate_table(left_col, right_col, table_title):
    
    # Do not use column headers
    # 不要使用列标题
    col_headers = None
    
    # Generate the right table
    # 生成右边的表
    if right_col:
        # Add padding
        # 添加填充
        if len(right_col) < len(left_col):
            right_col += [(' ', ' ')] * (len(left_col) - len(right_col))
        elif len(right_col) > len(left_col):
            left_col += [(' ', ' ')] * (len(right_col) - len(left_col))
        right_col = [('%-21s' % ('  '+k), v) for k,v in right_col]
        
        # Generate the right table
        # 生成右边的表
        gen_stubs_right, gen_data_right = zip_longest(*right_col)
        gen_table_right = SimpleTable(gen_data_right,
                                          col_headers,
                                          gen_stubs_right,
                                          title = table_title,
                                          txt_fmt = fmt_2cols)
    else:
        # If there is no right table set the right table to empty
        # 如果没有右边的表，将右边的表设置为空
        gen_table_right = []

    # Generate the left table  
    # 生成左边的表
    gen_stubs_left, gen_data_left = zip_longest(*left_col) 
    gen_table_left = SimpleTable(gen_data_left,
                                 col_headers,
                                 gen_stubs_left,
                                 title = table_title,
                                 txt_fmt = fmt_2cols)

    
    # Merge the left and right tables to make a single table
    # 合并左右表格以组成一个表格
    gen_table_left.extend_right(gen_table_right)
    general_table = gen_table_left

    return general_table


def get_env_param():
    
    # Create a simulation environment
    # 创建一个模拟环境
    env = sca.MarketEnvironment()

    # Set the title for the financial parameters table
    # 设置金融参数表的标题
    fp_title = 'Financial Parameters'

    # Get the default financial parameters from the simulation environment
    # 从模拟环境中获取默认金融参数
    fp_left_col = [('Annual Volatility:', ['{:.0f}%'.format(env.anv * 100)]),
                   ('Daily Volatility:', ['{:.1f}%'.format(env.dpv * 100)])]
    
    fp_right_col = [('Bid-Ask Spread:', ['{:.3f}'.format(env.basp)]),
                    ('Daily Trading Volume:', ['{:,.0f}'.format(env.dtv)])]

    # Set the title for the Almgren and Chriss Model parameters table
    # 设置Almgren和Chriss模型参数表的标题
    acp_title = 'Almgren and Chriss Model Parameters'

    # Get the default Almgren and Chriss Model Parameters from the simulation environment
    # 从模拟环境中获取默认的Almgren和Chriss模型参数
    acp_left_col = [('Total Number of Shares to Sell:', ['{:,}'.format(env.total_shares)]),
                    ('Starting Price per Share:', ['${:.2f}'.format(env.startingPrice)]),
                    ('Price Impact for Each 1% of Daily Volume Traded:', ['${}'.format(env.eta)]),                    
                    ('Number of Days to Sell All the Shares:', ['{}'.format(env.liquidation_time)]),
                    ('Number of Trades:', ['{}'.format(env.num_n)])]

    acp_right_col = [('Fixed Cost of Selling per Share:', ['${:.3f}'.format(env.epsilon)]),
                     ('Trader\'s Risk Aversion:', ['{}'.format(env.llambda)]),
                     ('Permanent Impact Constant:', ['{}'.format(env.gamma)]),
                     ('Single Step Variance:', ['{:.3f}'.format(env.singleStepVariance)]),
                     ('Time Interval between trades:', ['{}'.format(env.tau)])]

    # Generate tables with the default financial and AC Model parameters
    # 生成带有默认金融和AC模型参数的表
    fp_table = generate_table(fp_left_col, fp_right_col, fp_title)
    acp_table = generate_table(acp_left_col, acp_right_col, acp_title)

    return fp_table, acp_table


def plot_price_model(seed = 0, num_days = 1000):
    
    # Create a simulation environment
    # 创建一个模拟环境
    env = sca.MarketEnvironment()

    # Reset the enviroment with the given seed
    # 用给定的种子重置环境
    env.reset(seed)

    # Create an array to hold the daily stock price for the given number of days
    # 创建一个数组以保存指定天数内的每日股价
    price_hist = np.zeros(num_days)

    # Get the simulated stock price movement from the environment
    # 从环境中获取模拟的股价走势
    for i in range(num_days):
        _, _, _, info = env.step(i)    
        price_hist[i] = info.price
    
    # Print Average and Standard Deviation in Stock Price
    # 打印股票价格的平均值和标准偏差
    print('Average Stock Price: ${:,.2f}'.format(price_hist.mean()))
    print('Standard Deviation in Stock Price: ${:,.2f}'.format(price_hist.std()))
#     print('Standard Deviation of Random Noise: {:,.5f}'.format(np.sqrt(env.singleStepVariance * env.tau)))
    
    # Plot the price history for the given number of days
    # 绘制给定天数的价格历史记录
    price_df = pd.DataFrame(data = price_hist,  columns = ['Stock'], dtype = 'float64')
    ax = price_df.plot(colormap = 'cool', grid = False)
    ax.set_facecolor(color = 'k')
    ax = plt.gca()
    yNumFmt = mticker.StrMethodFormatter('${x:,.2f}')
    ax.yaxis.set_major_formatter(yNumFmt)
    plt.ylabel('Stock Price')
    plt.xlabel('days')
    plt.show()
    

    
def get_optimal_vals(lq_time = 60, nm_trades = 60, tr_risk = 1e-6, title = ''):
    
    # Create a simulation environment
    # 创建一个模拟环境
    env = sca.MarketEnvironment()

    # Reset the enviroment with the given parameters
    # 用给定的参数重置环境
    env.reset(liquid_time = lq_time, num_trades = nm_trades, lamb = tr_risk)

    # Set the title for the AC Optimal Strategy table
    # 设置AC最佳策略表的标题
    if title == '':
        title = 'AC Optimal Strategy'
    else:
        title = 'AC Optimal Strategy for ' + title

    # Get the AC optimal values from the environment
    # 从环境中获得AC的最佳值
    E = env.get_AC_expected_shortfall(env.total_shares)
    V = env.get_AC_variance(env.total_shares)
    U = env.compute_AC_utility(env.total_shares)

    left_col = [('Number of Days to Sell All the Shares:', ['{}'.format(env.liquidation_time)]),
                ('Half-Life of The Trade:', ['{:,.1f}'.format(1 / env.kappa)]),
                ('Utility:', ['${:,.2f}'.format(U)])]

    right_col = [('Initial Portfolio Value:', ['${:,.2f}'.format(env.total_shares * env.startingPrice)]),
                 ('Expected Shortfall:', ['${:,.2f}'.format(E)]),
                 ('Standard Deviation of Shortfall:', ['${:,.2f}'.format(np.sqrt(V))])]

    # Generate the table with the AC optimal values
    # 生成具有AC最佳值的表
    val_table = generate_table(left_col, right_col, title)

    return val_table


def get_min_param():
    
    # Get the minimum impact AC parameters
    # 获得影响最小的AC参数
    min_impact = get_optimal_vals(lq_time = 250, nm_trades = 250, tr_risk = 1e-17, title = 'Minimum Impact')
    
    # Get the minimum variance AC parameters
    # 获得最小方差的AC参数
    min_var = get_optimal_vals(lq_time = 1, nm_trades = 1, tr_risk = 0.0058, title = 'Minimum Variance')
    
    return min_impact, min_var
  
            
def get_crfs(trisk):
    
    # Create the annotation label
    # 创建注释标签
    tr_st = '{:.0e}'.format(trisk)   
    lnum = tr_st.split('e')[0]   
    lexp = tr_st.split('e')[1]
    if np.abs(np.int(lexp)) < 10:
        lexp = lexp.replace('0', '', 1)    
    an_st = '$\lambda = ' + lnum + ' \\times 10^{' + lexp + '}$'
    
    # Set the correction factors for the annotation label
    # 设置注释标签的校正因子
    if trisk >= 1e-7 and trisk <= 4e-7:
        xcrf = 0.94
        ycrf = 2.5
        scrf = 0.1
    elif trisk > 4e-7 and trisk <= 9e-7:
        xcrf = 0.9
        ycrf = 2.5
        scrf = 0.06
    elif trisk > 9e-7 and trisk <= 1e-6:
        xcrf = 0.85
        ycrf = 2.5
        scrf = 0.06
    elif trisk > 1e-6 and trisk < 2e-6:
        xcrf = 1.2
        ycrf = 2.5
        scrf = 0.06
    elif trisk >= 2e-6 and trisk < 3e-6:
        xcrf = 0.8
        ycrf = 2.5
        scrf = 0.06
    elif trisk >= 3e-6 and trisk < 4e-6:
        xcrf = 0.7
        ycrf = 2.5
        scrf = 0.08
    elif trisk >= 4e-6 and trisk < 7e-6:
        xcrf = 1.4
        ycrf = 2.0
        scrf = 0.08
    elif trisk >= 7e-6 and trisk <= 1e-5:
        xcrf = 4.5
        ycrf = 1.5
        scrf = 0.08
    elif trisk > 1e-5 and trisk <= 2e-5:
        xcrf = 7.0
        ycrf = 1.1
        scrf = 0.08
    elif trisk > 2e-5 and trisk <= 5e-5:
        xcrf = 12.
        ycrf = 1.1
        scrf = 0.08
    elif trisk > 5e-5 and trisk <= 1e-4:
        xcrf = 30
        ycrf = 0.99
        scrf = 0.08
    else:
        xcrf = 1
        ycrf = 1
        scrf = 0.08
    
    return an_st, xcrf, ycrf, scrf
    

def plot_efficient_frontier(tr_risk = 1e-6):
    
    # Create a simulation environment
    # 创建一个模拟环境
    env = sca.MarketEnvironment()
    
    # Reset the enviroment with the given trader's risk aversion
    # 用给定的交易者风险规避来重置环境
    env.reset(lamb = tr_risk)

    # Get the expected shortfall and corresponding variance for the given trader's risk aversion
    # 获取给定交易者的风险规避的预期缺口和相应的方差
    tr_E = env.get_AC_expected_shortfall(env.total_shares)
    tr_V = env.get_AC_variance(env.total_shares)
    
    # Create empty arrays to hold our values of E, V, and U
    # 创建空数组以保存我们的E，V和U值
    E = np.array([])
    V = np.array([])
    U = np.array([])
    
    # Set the number of plot points for our frontier
    # 设置边界点的数量
    num_points = 7000
    
    # Set the values of the trader's risk aversion to plot
    # 设置交易者的风险规避值以进行绘制
    lambdas = np.linspace(1e-7, 1e-4, num_points)
    
    # Calclate E, V, U for each value of llambda
    # 计算每个lambda值的E,V,U
    for llambda in lambdas:
        env.reset(lamb = llambda)
        E = np.append(E, env.get_AC_expected_shortfall(env.total_shares))
        V = np.append(V, env.get_AC_variance(env.total_shares))
        U = np.append(U, env.compute_AC_utility(env.total_shares))
        
    # Plot E vs V and use U for the colorbar  
    # 绘制E，V并使用U作为颜色条
    cm = plt.cm.get_cmap('gist_rainbow')    
    sc = plt.scatter(V, E, s = 20, c = U, cmap = cm)
    plt.colorbar(sc, label = 'AC Utility', format = mticker.StrMethodFormatter('${x:,.0f}'))
    ax = plt.gca()
    ax.set_facecolor('k')
    ymin = E.min() * 0.7
    ymax = E.max() * 1.1
    plt.ylim(ymin, ymax)
    yNumFmt = mticker.StrMethodFormatter('${x:,.0f}')
    xNumFmt = mticker.StrMethodFormatter('{x:,.0f}')
    ax.yaxis.set_major_formatter(yNumFmt)
    ax.xaxis.set_major_formatter(xNumFmt)
    plt.xlabel('Variance of Shortfall')
    plt.ylabel('Expected Shortfall')
    
    # Get the annotation label and the correction factors
    # 获取注释标签和校正因子
    an_st, xcrf, ycrf, scrf = get_crfs(tr_risk)
    
    # Plot the annotation in the above plot
    # 在上图中绘制注释
    plt.annotate(an_st, xy = (tr_V, tr_E), xytext = (tr_V * xcrf, tr_E  * ycrf), color = 'w', size = 'large', 
                 arrowprops = dict(facecolor = 'cyan', shrink = scrf, width = 3, headwidth = 10))
    plt.show()
    
    
def round_trade_list(trl):
    
    # Round the shares in the trading list
    # 四舍五入交易清单中的股票
    trl_rd = np.around(trl)
        
    # Rounding the number of shares in the trading list sometimes results in selling more or less
    # shares than we have available. We calculate the difference between to total number of shares
    # sold in the original trading list and the number of shares sold in the rounded list.
    # This difference will be used to correct for rounding errors. 
    # 四舍五入交易清单中的股票数量有时会导致卖出的股票或多或少。 我们计算原始交易列表中出售的股票总数与四舍五入列表中出售的股票数量之间的差额。
    # 此差异将用于校正舍入误差。
    res = np.around(trl.sum() - trl_rd.sum())
        
    # Correct the number of shares sold due to rounding errors if necessary
    # 如有必要，请更正由于四舍五入引起的售出股票数量
    if res != 0:
        idx = trl_rd.nonzero()[0][-1]      
        trl_rd[idx] += res
        
    return trl_rd

    
def plot_trade_list(lq_time = 60, nm_trades = 60, tr_risk = 1e-6, show_trl = False):
    
    # Create simulation environment
    # 创建模拟环境
    env = sca.MarketEnvironment()

    # Reset the environment with the given parameters
    # 使用给定的参数重置环境
    env.reset(liquid_time = lq_time, num_trades = nm_trades, lamb = tr_risk)

    # Get the trading list from the environment
    # 从环境中获取交易清单
    trade_list = env.get_trade_list()
    
    # Add a zero at the beginning of the trade list to indicate that at time 0 we don't sell any stocks
    # 在交易清单的开头添加一个零，以表示在时间0我们不出售任何股票
    new_trl = np.insert(trade_list, 0, 0)

    # We create a dataframe with the trading list and trading trajectory
    # 我们使用交易清单和交易轨迹创建一个数据框
    df = pd.DataFrame(data = list(range(nm_trades + 1)),  columns = ['Trade Number'], dtype = 'float64')
    df['Stocks Sold'] = new_trl
    df['Stocks Remaining'] = (np.ones(nm_trades + 1) * env.total_shares) - np.cumsum(new_trl)

    # Create a figure with 2 plots in 1 row
    # 在1行中创建包含2个图的图形
    fig, axes = plt.subplots(nrows = 1, ncols = 2)
    
    # Make a scatter plot of the trade list
    # 绘制散点图
    df.iloc[1:].plot.scatter(x = 'Trade Number', y = 'Stocks Sold', c = 'Stocks Sold', colormap = 'gist_rainbow',
                                                 alpha = 1, sharex = False, s = 50, colorbar = False, ax = axes[0])
    
    # Plot a line through the points of the scatter plot of the trade list
    # 通过交易清单散点图的点画一条线
    axes[0].plot(df['Trade Number'].iloc[1:], df['Stocks Sold'].iloc[1:], linewidth = 2.0, alpha = 0.5)
    axes[0].set_facecolor(color = 'k')
    yNumFmt = mticker.StrMethodFormatter('{x:,.0f}')
    axes[0].yaxis.set_major_formatter(yNumFmt)
    axes[0].set_title('Trading List')

    # Make a scatter plot of the number of stocks remaining after each trade
    # 绘制散点图，显示每次交易后剩余的股票数量
    df.plot.scatter(x = 'Trade Number', y = 'Stocks Remaining', c = 'Stocks Remaining', colormap = 'gist_rainbow',
                                                 alpha = 1, sharex = False, s = 50, colorbar = False, ax = axes[1])
    
    # Plot a line through the points of the scatter plot of the number of stocks remaining after each trade
    # 通过散点图的点画一条线，表示每次交易后剩余的股票数量
    axes[1].plot(df['Trade Number'], df['Stocks Remaining'], linewidth = 2.0, alpha = 0.5)
    axes[1].set_facecolor(color = 'k')
    yNumFmt = mticker.StrMethodFormatter('{x:,.0f}')
    axes[1].yaxis.set_major_formatter(yNumFmt)
    axes[1].set_title('Trading Trajectory')
    
    # Set the spacing between plots
    # 设置图之间的间隔
    plt.subplots_adjust(wspace = 0.4)
    plt.show()
    
    print('\nNumber of Shares Sold: {:,.0f}\n'.format(new_trl.sum()))
    
    if show_trl:
        
        # Since we are not selling fractional shares we round up the shares in the trading list
        # 由于我们不出售零碎股票，因此将交易清单中的股票四舍五入
        rd_trl = round_trade_list(new_trl)
#         rd_trl = new_trl

        # We create a dataframe with the modified trading list and trading trajectory
        # 我们使用修改后的交易清单和交易轨迹创建一个数据框
        df2 = pd.DataFrame(data = list(range(nm_trades + 1)),  columns = ['Trade Number'], dtype = 'float64')
        df2['Stocks Sold'] = rd_trl
        df2['Stocks Remaining'] = (np.ones(nm_trades + 1) * env.total_shares) - np.cumsum(rd_trl)

        return df2.style.hide_index().format({'Trade Number': '{:.0f}', 'Stocks Sold': '{:,.0f}', 'Stocks Remaining': '{:,.0f}'})
#         return df2.style.hide_index().format({'Trade Number': '{:.0f}', 'Stocks Sold': '{:e}', 'Stocks Remaining': '{:e}'})
    

def implement_trade_list(seed = 0, lq_time = 60, nm_trades = 60, tr_risk = 1e-6):
    
    # Create simulation environment
    # 创建一个模拟环境
    env = sca.MarketEnvironment()

    # Reset the environment with the given parameters
    # 使用给定参数重置模拟环境
    env.reset(seed = seed, liquid_time = lq_time, num_trades = nm_trades, lamb = tr_risk)

    # Get the trading list from the environment
    # 从环境中获取交易清单
    trl = env.get_trade_list()
    
    # Since we are not selling fractional shares we round up the shares in the trading list
    # 由于我们不出售零碎股票，因此将交易清单中的股票四舍五入
    trade_list = round_trade_list(trl)
 
    # set the environment to make transactions
    # 设置交易环境开始交易
    env.start_transactions()
    
    # Create an array to hold the impacted stock price
    # 创建一个数组以保存受影响的股价
    price_hist = np.array([])

    # Implement the trading list in our similation environment
    # 在我们的模拟环境中实施交易清单
    for trade in trade_list:
        
        # Convert the number of shares to sell in each trade into an action
        # 将每笔交易中要出售的股票数量转换为一个动作
        action = trade / env.shares_remaining
        
        # Take a step in the environment my selling the number of shares in the current trade
        # 在环境中迈出一步，我出售当前交易中的股票数量
        _, _, _, info = env.step(action)
        
        # Get the impacted price from the environment
        # 从环境中获取受影响的价格
        price_hist = np.append(price_hist, info.exec_price)
        
        # If all shares have been sold, stop making transactions and get the implementation sortfall
        # 如果所有股票都已售出，请停止进行交易并获得实施排序
        if info.done:
            print('Implementation Shortfall: ${:,.2f} \n'.format(info.implementation_shortfall))
            break

    # Plot the impacted price
    # 绘制受影响的价格
    price_df = pd.DataFrame(data = price_hist,  columns = ['Stock'], dtype = 'float64')
    ax = price_df.plot(colormap = 'cool', grid = False)
    ax.set_facecolor(color = 'k')
    ax.set_title('Impacted Stock Price')
    ax = plt.gca()
    yNumFmt = mticker.StrMethodFormatter('${x:,.2f}')
    ax.yaxis.set_major_formatter(yNumFmt)
    plt.plot(price_hist, 'o')
    plt.ylabel('Stock Price')
    plt.xlabel('Trade Number')
    plt.show()


def get_av_std(lq_time = 60, nm_trades = 60, tr_risk = 1e-6, trs = 100):
    
    # Create simulation environment
    # 创建一个模拟环境
    env = sca.MarketEnvironment()

    # Reset the enviroment
    # 重置环境
    env.reset(liquid_time = lq_time, num_trades = nm_trades, lamb = tr_risk)

    # Get the trading list
    # 获取交易清单
    trl = env.get_trade_list()

    # Since we are not selling fractional shares we round up the shares in the trading list
    # 由于我们不出售零碎股票，因此将交易清单中的股票四舍五入
    trade_list = round_trade_list(trl)

    # Set the initial shortfall to zero
    # 将初始缺口设置为零
    shortfall_hist = np.array([])

    for episode in range(trs):
        
        # Print current episode every 100 episodes
        # 每100集打印当前一集
        if (episode + 1) % 100 == 0:
            print('Episode [{}/{}]'.format(episode + 1, trs), end = '\r', flush = True)
        
        # Reset the enviroment
        # 重置环境
        env.reset(seed = episode, liquid_time = lq_time, num_trades = nm_trades, lamb = tr_risk)

        # set the environment to make transactions
        # 设置环境开始交易
        env.start_transactions()

        for trade in trade_list:
            action = trade / env.shares_remaining
            _, _, _, info = env.step(action)

            if info.done:
                shortfall_hist = np.append(shortfall_hist, info.implementation_shortfall)
                break

    print('Average Implementation Shortfall: ${:,.2f}'.format(shortfall_hist.mean()))
    print('Standard Deviation of the Implementation Shortfall: ${:,.2f}'.format(shortfall_hist.std()))
    
    plt.plot(shortfall_hist, 'cyan', label='')
    plt.xlim(0, trs)
    ax = plt.gca()
    ax.set_facecolor('k')
    ax.set_xlabel('Episode', fontsize = 15)
    ax.set_ylabel('Implementation Shortfall (US $)', fontsize = 15)
    ax.axhline(shortfall_hist.mean(),0, 1, color = 'm', label='Average')
    yNumFmt = mticker.StrMethodFormatter('${x:,.0f}')
    ax.yaxis.set_major_formatter(yNumFmt)
    plt.legend()
    plt.show