
from sklearn.decomposition import PCA

import pandas as pd


# pass in the dataframe of stocks price by timestamp (col by row)
def getTopTwoPrincipleComponents(data):
    returns = data.pct_change()
    print("returns shape: " + str(returns.shape))
    # df (timeseries x stocks)
    df = returns.iloc[1:]
    print("df shape: " + str(df.shape))
    # transpose df to be able to get df (stock x timestamps)
    df_transposed = df.T
    print("df_transposed shape: " + str(df_transposed.shape))
    # get the index of the df
    index = df_transposed.index
    # get the principle components of the df
    pca = PCA()
    pca.fit(df_transposed)
    principle_components = pca.components_
    print("priciple_components shape: " + str(principle_components.shape))
    # dot product to reduce the dimension 
    # comp @ df.values = stock x stock
    result = principle_components@df.values 
    print("result shape: " + str(result.shape))
    # get the top 2 components
    points = result[0:2,:].T
    print("data points shape: " + str(points.shape))
    return pd.DataFrame(data=points, index = index, columns = ['x', 'y'])



    # calculate the sharpe ratio for a portfolio
def portfolio_sharpe_ratio(df, weights, adjusted):
    result_df = pd.DataFrame()
    portf_val = pd.DataFrame()
    stocks = df.columns
    for stock in stocks:
        temp = pd.DataFrame(df[stock])
        temp['Norm return '+stock] = temp / temp.iloc[0]
        temp['Allocation '+stock] = temp['Norm return '+stock] * weights[stock]
        portf_val[stock+' Position'] = temp['Allocation '+stock] * 10000
    
    portf_val['Total Position'] = portf_val.sum(axis=1)
    portf_val['Daily Return'] = portf_val['Total Position'].pct_change(1)
    Sharpe_ratio = portf_val['Daily Return'].mean() / portf_val['Daily Return'].std()
    if adjusted:
        Sharpe_ratio = (252**0.5) * Sharpe_ratio
    return Sharpe_ratio


def rand_weights(n):
    ''' Produces n random weights that sum to 1 '''
    k = np.random.rand(n)
    return k / sum(k)


def random_portfolio(returns):
    ''' 
    Returns the mean and standard deviation of returns for a random portfolio
    '''

    p = np.asmatrix(np.mean(returns, axis=1))
    w = np.asmatrix(rand_weights(returns.shape[0]))
    C = np.asmatrix(np.cov(returns))
    
    mu = w * p.T
    sigma = np.sqrt(w * C * w.T)
    
    # This recursion reduces outliers to keep plots pretty
    if sigma > 2:
        return random_portfolio(returns)
    return mu, sigma



def optimal_portfolio(returns):
    n = len(returns)
    returns = np.asmatrix(returns)
    
    N = 500
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]
    
    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))
    
    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] 
                  for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks
