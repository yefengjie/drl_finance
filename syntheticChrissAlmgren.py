import random
import numpy as np
import collections


# ------------------------------------------------ Financial Parameters --------------------------------------------------- #
# ------------------------------------------------ 金融参数 --------------------------------------------------- #

ANNUAL_VOLAT = 0.12                                # Annual volatility in stock price 股价年度波动
BID_ASK_SP = 1 / 8                                 # Bid-ask spread 买卖差价
DAILY_TRADE_VOL = 5e6                              # Average Daily trading volume 平均每日交易量
TRAD_DAYS = 250                                    # Number of trading days in a year 一年中的交易日数
DAILY_VOLAT = ANNUAL_VOLAT / np.sqrt(TRAD_DAYS)    # Daily volatility in stock price 股价的每日波动


# ----------------------------- Parameters for the Almgren and Chriss Optimal Execution Model ----------------------------- #
# ----------------------------- Almgren和Chriss最佳执行模型的参数 ----------------------------- #

TOTAL_SHARES = 1000000                                               # Total number of shares to sell 出售股份总数
STARTING_PRICE = 50                                                  # Starting price per share 每股起拍价
LLAMBDA = 1e-6                                                       # Trader's risk aversion 交易者的风险规避
LIQUIDATION_TIME = 60                                                # How many days to sell all the shares. 多少天卖完所有股票
NUM_N = 60                                                           # Number of trades 交易数量
EPSILON = BID_ASK_SP / 2                                             # Fixed Cost of Selling. 固定销售成本
SINGLE_STEP_VARIANCE = (DAILY_VOLAT  * STARTING_PRICE) ** 2          # Calculate single step variance 计算单步方差
ETA = BID_ASK_SP / (0.01 * DAILY_TRADE_VOL)                          # Price Impact for Each 1% of Daily Volume Traded 每日交易量的1%的价格影响
GAMMA = BID_ASK_SP / (0.1 * DAILY_TRADE_VOL)                         # Permanent Impact Constant 永久影响常数

# ----------------------------------------------------------------------------------------------------------------------- #


# Simulation Environment
# 模拟环境

class MarketEnvironment():
    
    def __init__(self, randomSeed = 0,
                 lqd_time = LIQUIDATION_TIME,
                 num_tr = NUM_N,
                 lambd = LLAMBDA):
        
        # Set the random seed
        # 设置随机种子
        random.seed(randomSeed)
        
        # Initialize the financial parameters so we can access them later
        # 初始化金融参数，以便稍后使用
        self.anv = ANNUAL_VOLAT
        self.basp = BID_ASK_SP
        self.dtv = DAILY_TRADE_VOL
        self.dpv = DAILY_VOLAT
        
        # Initialize the Almgren-Chriss parameters so we can access them later
        # 初始化Almgren-Chriss参数，以便我们以后可以使用它们
        self.total_shares = TOTAL_SHARES
        self.startingPrice = STARTING_PRICE
        self.llambda = lambd
        self.liquidation_time = lqd_time
        self.num_n = num_tr
        self.epsilon = EPSILON
        self.singleStepVariance = SINGLE_STEP_VARIANCE
        self.eta = ETA
        self.gamma = GAMMA
        
        # Calculate some Almgren-Chriss parameters
        # 计算一些Almgren-Chriss参数
        self.tau = self.liquidation_time / self.num_n 
        self.eta_hat = self.eta - (0.5 * self.gamma * self.tau)
        self.kappa_hat = np.sqrt((self.llambda * self.singleStepVariance) / self.eta_hat)
        self.kappa = np.arccosh((((self.kappa_hat ** 2) * (self.tau ** 2)) / 2) + 1) / self.tau

        # Set the variables for the initial state
        # 设置初始状态的变量
        self.shares_remaining = self.total_shares
        self.timeHorizon = self.num_n
        self.logReturns = collections.deque(np.zeros(6))
        
        # Set the initial impacted price to the starting price
        # 将初始受影响的价格设置为起始价格
        self.prevImpactedPrice = self.startingPrice

        # Set the initial transaction state to False
        # 将初始交易状态设置为False
        self.transacting = False
        
        # Set a variable to keep trak of the trade number
        # 设置变量以跟踪交易号
        self.k = 0
        
        
    def reset(self, seed = 0, liquid_time = LIQUIDATION_TIME, num_trades = NUM_N, lamb = LLAMBDA):
        
        # Initialize the environment with the given parameters
        # 使用给定的参数初始化环境
        self.__init__(randomSeed = seed, lqd_time = liquid_time, num_tr = num_trades, lambd = lamb)
        
        # Set the initial state to [0,0,0,0,0,0,1,1]
        # 设置初始状态为[0,0,0,0,0,0,1,1]
        self.initial_state = np.array(list(self.logReturns) + [self.timeHorizon / self.num_n, \
                                                               self.shares_remaining / self.total_shares])
        return self.initial_state

    
    def start_transactions(self):
        
        # Set transactions on
        # 设置交易开启
        self.transacting = True
        
        # Set the minimum number of stocks one can sell
        # 设置一个可以卖出的最小股票数量
        self.tolerance = 1
        
        # Set the initial capture to zero
        # 将初始捕获设置为零
        self.totalCapture = 0
        
        # Set the initial previous price to the starting price
        # 将初始的先前价格设置为起始价格
        self.prevPrice = self.startingPrice
        
        # Set the initial square of the shares to sell to zero
        # 将要出售的股票的初始平方设为零
        self.totalSSSQ = 0
        
        # Set the initial square of the remaing shares to sell to zero
        # 将剩余要出售的股票的初始平方设为零
        self.totalSRSQ = 0
        
        # Set the initial AC utility
        # 设置初始AC实用程序
        self.prevUtility = self.compute_AC_utility(self.total_shares)
        

    def step(self, action):
        
        # Create a class that will be used to keep track of information about the transaction
        # 创建一个类，用于跟踪有关交易的信息
        class Info(object):
            pass        
        info = Info()
        
        # Set the done flag to False. This indicates that we haven't sold all the shares yet.
        # 将完成标志设置为False。 这表明我们尚未出售所有股票。
        info.done = False
                
        # During training, if the DDPG fails to sell all the stocks before the given 
        # number of trades or if the total number shares remaining is less than 1, then stop transacting,
        # set the done Flag to True, return the current implementation shortfall, and give a negative reward.
        # The negative reward is given in the else statement below.
        # 在训练过程中，如果DDPG在给定交易次数之前未能卖出所有股票，或者剩余总股数少于1，则停止交易，将完成的Flag设置为True，返回当前的实现缺口，并给出 负面奖励。负奖励在下面的else语句中给出。
        if self.transacting and (self.timeHorizon == 0 or abs(self.shares_remaining) < self.tolerance):
            self.transacting = False
            info.done = True
            info.implementation_shortfall = self.total_shares * self.startingPrice - self.totalCapture
            info.expected_shortfall = self.get_expected_shortfall(self.total_shares)
            info.expected_variance = self.singleStepVariance * self.tau * self.totalSRSQ
            info.utility = info.expected_shortfall + self.llambda * info.expected_variance
            
        # We don't add noise before the first trade  
        # 我们不会在第一次交易前增加噪音
        if self.k == 0:
            info.price = self.prevImpactedPrice
        else:
            # Calculate the current stock price using arithmetic brownian motion
            # 使用算术布朗运动计算当前股价
            info.price = self.prevImpactedPrice + np.sqrt(self.singleStepVariance * self.tau) * random.normalvariate(0, 1)
      
        # If we are transacting, the stock price is affected by the number of shares we sell. The price evolves 
        # according to the Almgren and Chriss price dynamics model. 
        # 如果我们进行交易，则股价会受到我们出售股票数量的影响。 价格根据Almgren和Chriss的价格动态模型发展。
        if self.transacting:
            
            # If action is an ndarray then extract the number from the array
            # 如果action是ndarray，则从数组中提取数字
            if isinstance(action, np.ndarray):
                action = action.item()            

            # Convert the action to the number of shares to sell in the current step
            # 将动作转换为当前步骤中要出售的股票数量
            sharesToSellNow = self.shares_remaining * action
            # sharesToSellNow = min(self.shares_remaining * action, self.shares_remaining)
    
            if self.timeHorizon < 2:
                sharesToSellNow = self.shares_remaining

            # Since we are not selling fractions of shares, round up the total number of shares to sell to the nearest integer.
            # 由于我们不出售零碎股份，因此将要出售的股份总数四舍五入到最接近的整数。
            info.share_to_sell_now = np.around(sharesToSellNow)

            # Calculate the permanent and temporary impact on the stock price according the AC price dynamics model
            # 根据AC价格动态模型计算对股票价格的永久和暂时影响
            info.currentPermanentImpact = self.permanentImpact(info.share_to_sell_now)
            info.currentTemporaryImpact = self.temporaryImpact(info.share_to_sell_now)
                
            # Apply the temporary impact on the current stock price    
            # 对当前股价施加暂时影响
            info.exec_price = info.price - info.currentTemporaryImpact
            
            # Calculate the current total capture
            # 计算当前的总捕获量
            self.totalCapture += info.share_to_sell_now * info.exec_price

            # Calculate the log return for the current step and save it in the logReturn deque
            # 计算当前步骤的log返回值并将其保存在logReturn双端队列中
            self.logReturns.append(np.log(info.price/self.prevPrice))
            self.logReturns.popleft()
            
            # Update the number of shares remaining
            # 更新剩余股份数
            self.shares_remaining -= info.share_to_sell_now
            
            # Calculate the runnig total of the squares of shares sold and shares remaining
            # 计算已售股和剩余股的平方的总和
            self.totalSSSQ += info.share_to_sell_now ** 2
            self.totalSRSQ += self.shares_remaining ** 2
                                        
            # Update the variables required for the next step
            # 更新下一步所需的变量
            self.timeHorizon -= 1
            self.prevPrice = info.price
            self.prevImpactedPrice = info.price - info.currentPermanentImpact
            
            # Calculate the reward
            # 计算奖励
            currentUtility = self.compute_AC_utility(self.shares_remaining)
            reward = (abs(self.prevUtility) - abs(currentUtility)) / abs(self.prevUtility)
            self.prevUtility = currentUtility
            
            # If all the shares have been sold calculate E, V, and U, and give a positive reward.
            # 如果所有股份均已售出，请计算E，V和U，并给予正数奖励。
            if self.shares_remaining <= 0:
                
                # Calculate the implementation shortfall
                # 计算实施缺口
                info.implementation_shortfall  = self.total_shares * self.startingPrice - self.totalCapture
                   
                # Set the done flag to True. This indicates that we have sold all the shares
                # 将完成标志设置为True。 这表明我们已经卖出了所有股份
                info.done = True
        else:
            reward = 0.0
        
        self.k += 1
            
        # Set the new state
        # 设置新状态
        state = np.array(list(self.logReturns) + [self.timeHorizon / self.num_n, self.shares_remaining / self.total_shares])

        return (state, np.array([reward]), info.done, info)

   
    def permanentImpact(self, sharesToSell):
        # Calculate the permanent impact according to equations (6) and (1) of the AC paper
        # 根据AC paper的方程式（6）和（1）计算永久冲击
        pi = self.gamma * sharesToSell
        return pi

    
    def temporaryImpact(self, sharesToSell):
        # Calculate the temporary impact according to equation (7) of the AC paper
        # 根据AC paper的方程式（7）计算临时影响
        ti = (self.epsilon * np.sign(sharesToSell)) + ((self.eta / self.tau) * sharesToSell)
        return ti
    
    def get_expected_shortfall(self, sharesToSell):
        # Calculate the expected shortfall according to equation (8) of the AC paper
        # 根据ACpaper的公式（8）计算预期的不足
        ft = 0.5 * self.gamma * (sharesToSell ** 2)        
        st = self.epsilon * sharesToSell
        tt = (self.eta_hat / self.tau) * self.totalSSSQ
        return ft + st + tt

    
    def get_AC_expected_shortfall(self, sharesToSell):
        # Calculate the expected shortfall for the optimal strategy according to equation (20) of the AC paper
        # 根据AC paper的等式（20）计算最佳策略的预期缺口
        ft = 0.5 * self.gamma * (sharesToSell ** 2)        
        st = self.epsilon * sharesToSell        
        tt = self.eta_hat * (sharesToSell ** 2)       
        nft = np.tanh(0.5 * self.kappa * self.tau) * (self.tau * np.sinh(2 * self.kappa * self.liquidation_time) \
                                                      + 2 * self.liquidation_time * np.sinh(self.kappa * self.tau))       
        dft = 2 * (self.tau ** 2) * (np.sinh(self.kappa * self.liquidation_time) ** 2)   
        fot = nft / dft       
        return ft + st + (tt * fot)  
        
    
    def get_AC_variance(self, sharesToSell):
        # Calculate the variance for the optimal strategy according to equation (20) of the AC paper
        # 根据AC paper的方程（20）计算最佳策略的方差
        ft = 0.5 * (self.singleStepVariance) * (sharesToSell ** 2)                        
        nst  = self.tau * np.sinh(self.kappa * self.liquidation_time) * np.cosh(self.kappa * (self.liquidation_time - self.tau)) \
               - self.liquidation_time * np.sinh(self.kappa * self.tau)        
        dst = (np.sinh(self.kappa * self.liquidation_time) ** 2) * np.sinh(self.kappa * self.tau)        
        st = nst / dst
        return ft * st
        
        
    def compute_AC_utility(self, sharesToSell):    
        # Calculate the AC Utility according to pg. 13 of the AC paper
        # 根据AC paper的13页计算AC效用
        if self.liquidation_time == 0:
            return 0        
        E = self.get_AC_expected_shortfall(sharesToSell)
        V = self.get_AC_variance(sharesToSell)
        return E + self.llambda * V
    
    
    def get_trade_list(self):
        # Calculate the trade list for the optimal strategy according to equation (18) of the AC paper
        # 根据AC paper的公式（18）计算最佳策略的交易清单
        trade_list = np.zeros(self.num_n)
        ftn = 2 * np.sinh(0.5 * self.kappa * self.tau)
        ftd = np.sinh(self.kappa * self.liquidation_time)
        ft = (ftn / ftd) * self.total_shares
        for i in range(1, self.num_n + 1):       
            st = np.cosh(self.kappa * (self.liquidation_time - (i - 0.5) * self.tau))
            trade_list[i - 1] = st
        trade_list *= ft
        return trade_list
     
        
    def observation_space_dimension(self):
        # Return the dimension of the state
        # 返回状态的维数
        return 8
    
    
    def action_space_dimension(self):
        # Return the dimension of the action
        # 返回动作的维数
        return 1
    
    
    def stop_transactions(self):
        # Stop transacting
        # 停止交易
        self.transacting = False            
            
           