import wrangle as w
  
import pandas as pd
import numpy as np

import sklearn.metrics as metric
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures

########################################### BASELINE PREDICTIONS


def model_comparisson(X_train, y_train, X_validate, y_validate):
    train_predictions = y_train.copy()
    validate_predictions = y_validate.copy()

    # create the metric_df as a blank dataframe
    metric_df = pd.DataFrame() 

    #OLS Model
    lm = LinearRegression(normalize=True)
    lm.fit(X_train, y_train)
    train_predictions['lm'] = lm.predict(X_train)
    # predict validate
    validate_predictions['lm'] = lm.predict(X_validate)
    metric_df = make_metric_df(y_train, train_predictions.lm, y_validate, validate_predictions.lm, metric_df, model_name = 'OLS Regressor')

    #Lasso Lars
    # create the model object
    lars = LassoLars(alpha=1)
    lars.fit(X_train, y_train)
    # predict train
    train_predictions['lars'] = lars.predict(X_train)
    # predict validate
    validate_predictions['lars'] = lars.predict(X_validate)
    metric_df = make_metric_df(y_train, train_predictions.lars, y_validate, validate_predictions.lars, metric_df, model_name = 'Lasso_alpha_1')
    
    #Tweedie Regressor/GLM
    # create the model object
    glm = TweedieRegressor(power=1, alpha=0)
    glm.fit(X_train, y_train)
    # predict train
    train_predictions['glm'] = glm.predict(X_train)
    # predict validate
    validate_predictions['glm'] = glm.predict(X_validate)
    metric_df = make_metric_df(y_train, train_predictions.glm, y_validate, validate_predictions.glm, metric_df, model_name = 'GLM')

    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2)
    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(X_train)
    # transform X_validate_scaled & X_test_scaled
    X_validate_degree2 = pf.transform(X_validate)
    # create the model object
    lm2 = LinearRegression(normalize=True)
    lm2.fit(X_train_degree2, y_train)
    # predict train
    train_predictions['poly_2'] = lm2.predict(X_train_degree2)
    # predict validate
    validate_predictions['poly_2'] = lm2.predict(X_validate_degree2)
    metric_df = make_metric_df(y_train, train_predictions.poly_2, y_validate, validate_predictions.poly_2, metric_df, model_name = 'Polynomial')

    return metric_df

def make_metric_df(y_train, y_train_pred, y_validate, y_validate_pred,  metric_df,model_name ):
    if metric_df.size ==0:
        metric_df = pd.DataFrame(data=[
            {
                'model': model_name, 
                f'RMSE_train': metric.mean_squared_error(
                    y_train,
                    y_train_pred) ** .5,
                f'r^2_train': metric.explained_variance_score(
                    y_train,
                    y_train_pred),
                f'RMSE_validate': metric.mean_squared_error(
                    y_validate,
                    y_validate_pred) ** .5,
                f'r^2_validate': metric.explained_variance_score(
                    y_validate,
                    y_validate_pred)
            }])
        return metric_df
    else:
        return metric_df.append(
            {
                'model': model_name, 
                f'RMSE_train': metric.mean_squared_error(
                    y_train,
                    y_train_pred) ** .5,
                f'r^2_train': metric.explained_variance_score(
                    y_train,
                    y_train_pred),
                f'RMSE_validate': metric.mean_squared_error(
                    y_validate,
                    y_validate_pred) ** .5,
                f'r^2_validate': metric.explained_variance_score(
                    y_validate,
                    y_validate_pred)
            }, ignore_index=True)

def baseline_models(y_train, y_validate):
    
    train_predictions = y_train.copy()
    validate_predictions = y_validate.copy()
    
    y_pred_mean = y_train.property_value.mean()
    train_predictions['y_pred_mean'] = y_pred_mean
    validate_predictions['y_pred_mean'] = y_pred_mean
    
    y_pred_median = y_train.property_value.median()
    train_predictions['y_pred_median'] = y_pred_median
    validate_predictions['y_pred_median'] = y_pred_median

    # create the metric_df as a blank dataframe
    metric_df = pd.DataFrame(data=[
    {
        'model': 'mean_baseline', 
        'RMSE_train': metric.mean_squared_error(
            y_train,
            train_predictions['y_pred_mean']) ** .5,
        'RMSE_validate': metric.mean_squared_error(
            y_validate,
            validate_predictions['y_pred_mean']) ** .5,
        'Difference': (( metric.mean_squared_error(
            y_train,
            train_predictions['y_pred_mean']) ** .5)-(metric.mean_squared_error(
            y_validate,
            validate_predictions['y_pred_mean']) ** .5))
    }])

    return metric_df.append(
            {
                'model': 'median_baseline', 
                'RMSE_train': metric.mean_squared_error(
                    y_train,
                    train_predictions['y_pred_median']) ** .5,
                'RMSE_validate': metric.mean_squared_error(
                    y_validate,
                    validate_predictions['y_pred_median']) ** .5,
                'Difference': (( metric.mean_squared_error(
                    y_train,
                    train_predictions['y_pred_median']) ** .5)-(metric.mean_squared_error(
                    y_validate,
                    validate_predictions['y_pred_median']) ** .5))
            }, ignore_index=True)

###############################
def X_overwrite(X_train, X_validate, X_test):
    
    X_train=X_train.drop(columns=['county_LA', 'year_built', 'bed_rooms', 'county_Ventura', 'property_age', 'property_square_feet', 'county_Orange', 'house_to_lot_sqft_ratio'])
    
    X_validate=X_validate.drop(columns=['county_LA', 'year_built', 'bed_rooms', 'county_Ventura', 'property_age', 'property_square_feet', 'county_Orange', 'house_to_lot_sqft_ratio'])
    
    X_test=X_test.drop(columns=['county_LA', 'year_built', 'bed_rooms', 'county_Ventura', 'property_age', 'property_square_feet', 'county_Orange', 'house_to_lot_sqft_ratio'])
    
    return X_train, X_validate, X_test
    
    
    
