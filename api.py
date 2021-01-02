import time
from flask import Flask
from pandas_datareader import data
from datetime import datetime
import json
import pandas as pd

# from pypfopt import expected_returns

from utils import getTopTwoPrincipleComponents




options = ['BA', 'GOOGL', 'AMZN', 'DIS', 'AAPL']
start = datetime(2015, 1, 1)
end = datetime.now()

stock_df = data.DataReader(options, data_source='yahoo', start = start, end=end)
stock_df.index = [str(x) for x in stock_df.index]
stock_df_adj_close = stock_df['Adj Close']


# compute the covariance
# covariance = risk_models.CovarianceShrinkage(stock_df).ledoit_wolf()
# You don't have to provide expected returns in this case
# ef_min_vol = EfficientFrontier(None, covariance, weight_bounds=(0, None))
# ef_min_vol.min_volatility()
# # ef.max_sharpe()
# weights_min_vol = ef_min_vol.clean_weights()
# pie_stocks_min_vol = list(weights_min_vol.keys())



app = Flask(__name__)


@app.route('/stocks_data')
def get_stock_data():
    json_string = stock_df_adj_close.to_json(orient='index')
    json_stocks = json.loads(json_string)
    return json_stocks


@app.route('/correlation_matrix')
def get_correlation_matrix():
    # This returns a list of symbols, from least to most correlated
    sum_corr = stock_df_adj_close.corr().sum().sort_values(ascending=False).index.values
    # put in decending order and get the correlations
    correlation_df = stock_df_adj_close[sum_corr].corr()
    correlation_json_string = correlation_df.to_json(orient='index')
    correlation_parsed = json.loads(correlation_json_string)
    return correlation_parsed


@app.route('/top_principle_compontents')
def get_top_two_principle_components():
    top_comp = getTopTwoPrincipleComponents(stock_df_adj_close)
    top_comp_json = top_comp.to_json(orient='index')
    top_comp_parsed = json.loads(top_comp_json)
    return top_comp_parsed



# find expected returns
# @app.route('/expected_returns')
# def get_expected_returns():
#     exp_returns = expected_returns.capm_return(stock_df_adj_close)
#     exp_df = pd.DataFrame(exp_returns.index)
#     exp_df['Expected Returns'] = [round(exp_r*100, 2) for exp_r in exp_returns.values]
#     exp_df_json = exp_df.to_json(orient='index')
#     exp_df_parsed = json.loads(exp_df_json)
#     return exp_df_parsed



# # get the mean variance optimization portfolio
# def get_min_variance_pie_plot():
#     ## NEED TO CONVERT TO JSON
#     # fig_hbar_vol = go.Figure(go.Bar(x=list(weights.values()), y=list(weights.keys()), orientation='h'))
#     fig_min_vol_pie = go.Figure(go.Pie(labels=list(pie_stocks), values=[weights[s] for s in pie_stocks],  textinfo='label+percent'))
#     # get the annual vol for the minimized vol portfolio
#     annual_vol = ef.portfolio_performance()


#     ## NEED TO MAKE SEPARATE FUNCTION FOR ALL OF THIS
#     min_vol_sharpe = portfolio_sharpe_ratio(stock_df, weights, True)
#     vol = html.Div([
#         html.H2("GMV Performance: "),
#         html.P("Expected annual volatility: " + str(round(annual_vol[1]*100, 2))),
#         html.P("Sharpe ratio: " + str(round(min_vol_sharpe, 2)))
#         ], style={'color': colors['text'], 'fontSize': 18})
