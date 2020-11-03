# Title: functions, gdp loss
# Author: Doug Hart
# Date Created: 11/2/2020
# Last Updated: 11/2/2020

import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

def run_arima(series, date, p,d,q):
    '''should run ARIMA regression on specified series
    need to add returns for fit statistics for comparison
    series: column of df to forecast
    dates: date column, as multi-index doesn't seem compatible here
    
    The (p,d,q) order of the model for the number of AR parameters,
    differences, and MA parameters to use.
    '''
    model = ARIMA(series, dates = date, order=(p, d, q)).fit()
    
    fig, ax = plt.subplots(1, figsize=(14, 4))
    ax.plot(series.index, series)
    fig = model.plot_predict('2020-1-1', '2021', 
                                  dynamic=True, ax=ax, plot_insample=False)
    
    print("ARIMA(1, 1, 5) coefficients from first model:\n  Intercept {0:2.2f}\n  AR {1}".format(
    model.params[0], 
        format_list_of_floats(list(model.params[1:]))
    ))
    
    return model

def evaluate_models(dataset, timevar, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        print('*')
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse , aic, bic = evaluate_arima_model(dataset, timevar, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.3f AIC%s BIC%s' % (order,mse, aic, bic))
                except:
                    continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))


def evaluate_arima_model(X, timevar, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, dates= timevar, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    bic = model.bic
    aic = modle.aic
    return error, aic, bic

