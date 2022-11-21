import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns 

import sklearn.preprocessing as pre
from sklearn.model_selection import train_test_split

import env

        ####################################################################################
                                    # ACQUIRE DATA #
#####################################################################################

def get_db_url(db, user = env.user, password = env.password, host = env.host):
    ''' This function takes in the name of a database, and imported 
            username, password, and host from an env file and returns
            the url that accesses that database'''
    
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
            
        
    
        
def new_zillow_data():
    ''' 
    This function uses a SQL query to select desired features from the zillow dataset 
    from the Codeup database and returns them in a dataframe
    '''
    
    sql_query = '''
        SELECT bathroomcnt,
            bedroomcnt,
            taxvaluedollarcnt,
            calculatedfinishedsquarefeet,
            yearbuilt,
            fips,
            lotsizesquarefeet
        FROM properties_2017 
        JOIN predictions_2017 USING (parcelid)
        JOIN propertylandusetype USING (propertylandusetypeid)
        WHERE propertylandusedesc IN ('Single Family Residential' , 'Inferred Single Family Residential')
                AND YEAR(transactiondate) = 2017;
                '''
    df = pd.read_sql(sql_query, get_db_url(db= 'zillow'))
    
    return df


def acquire_zillow_data(new = False):
    ''' 
    Checks to see if there is a local copy of the data, 
    if not or if new = True then go get data from Codeup database
    '''
    
    filename = 'zillow.csv'
    
    #if we don't have cached data or we want to get new data go get it from server
    if (os.path.isfile(filename) == False) or (new == True):
        df = new_zillow_data()
        #save as csv
        df.to_csv(filename,index=False)

    #else used cached data
    else:
        df = pd.read_csv(filename)
          
    return df


#####################################################################################
                               # CLEAN DATA #
#####################################################################################

def clean_data(df):
    
    #create column to calculate the age of the property
    df['property_age'] = 2017 - df['yearbuilt']
    
    #rename columns
    df = df.rename(columns = {'bedroomcnt':'bed_rooms', 
                          'bathroomcnt':'bath_rooms',
                          'calculatedfinishedsquarefeet':'house_square_feet',
                          'taxvaluedollarcnt':'property_value', 
                          'lotsizesquarefeet':'property_square_feet'}) 
    
    # replace whitespace with null value
    df = df.replace(r'^\s*$', np.nan, regex=True)
    
    # drop null values
    df = df.dropna()
    
    
    # change datatypes
    df["year_built"] = df["yearbuilt"].astype(int)
    df["bed_rooms"] = df["bed_rooms"].astype(int)  
    df["bath_rooms"] = df["bath_rooms"].astype(int) 
    df["house_square_feet"] = df["house_square_feet"].astype(int)
    df["property_age"] = df["property_age"].astype(int)
    df["property_square_feet"] = df["property_square_feet"].astype(int)
    
    # Killing off outliers
    df=df[df.bed_rooms <=5]
    df=df[df.bed_rooms >=1]
    df=df[df.bath_rooms <=5]
    df=df[df.house_square_feet <= 7000]
    df=df[df.house_square_feet >= 1000]
    df=df[df.property_value <= 1300000]    
    df=df[df.property_square_feet <= 20000]
    
    # rename fips to county names
    df['county'] = df.fips.replace({6037:'LA', 6059:'Orange', 6111:'Ventura'})
    
    #get dummies for counties
    dummy_df = pd.get_dummies(df[['county']], dummy_na=False)
    df = pd.concat([df, dummy_df], axis=1)
    
    # drop unwanted columns
    df = df.drop(columns = ["fips", "yearbuilt"])
    # using .loc to delete all rows where bath_room = 0
    df = df.loc[df["bath_rooms"] !=0]
    
    return df



###################################################################################
                                    #SPLIT DATA#
###################################################################################

def split_data(df):
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=1989)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=1989)
    return train, validate, test

#############################################   SECONDARY SPLIT

def X_y_split(df, target):
    
    train, validate, test = split_data(df)
    
    X_cols = ['bed_rooms', 
              'bath_rooms', 
              'year_built',
              'property_age',
              'county_Orange', 
              'county_Ventura',
              'county_LA',
              'house_square_feet',
              'property_square_feet']

    target = 'property_value'
    
    X_train = train[X_cols]
    y_train = train[target]

    X_validate = validate[X_cols]
    y_validate = validate[target]

    X_test = test[X_cols]
    y_test = test[target]
    
    #forcing y's as series into dataframes
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    y_test = pd.DataFrame(y_test)
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test

###################################################################################
                               # FEATURE ENGINEERING #
###################################################################################

############## THE FOLLOWING 3 FUNCTIONS GENERATE FEATURES FOR TRAIN, VALIDATE, AND TEST

def bed_to_bath_generator(train, validate, test):
    train["bath_to_bed_ratio"] = train.bath_rooms/train.bed_rooms
    validate["bath_to_bed_ratio"] = validate.bath_rooms/validate.bed_rooms
    test["bath_to_bed_ratio"] = test.bath_rooms/test.bed_rooms
    
    return train, validate, test

def house_to_lot_generator(train, validate, test):
    train["house_to_lot_sqft_ratio"]=train.house_square_feet/train.property_square_feet
    validate["house_to_lot_sqft_ratio"]=validate.house_square_feet/validate.property_square_feet
    test["house_to_lot_sqft_ratio"]=test.house_square_feet/test.property_square_feet
    
    return train, validate, test

def bath_to_house_generator(train, validate, test):
    train["bath_to_house_sqft_ratio"]=train.bath_rooms/train.house_square_feet
    validate["bath_to_house_sqft_ratio"]=validate.bath_rooms/validate.house_square_feet
    test["bath_to_house_sqft_ratio"]=test.bath_rooms/test.house_square_feet
    
    return train, validate, test

############## COMPILATION OF FEATURE GENERATOR FUNCTIONS
def feature_generator(train, validate, test):
    bh_train, bh_validate, bh_test = bath_to_house_generator(train, validate, test)
    
    hl_train, hl_validate, hl_test = house_to_lot_generator(bh_train, bh_validate, bh_test)
    
    bb_train, bb_validate, bb_test = bed_to_bath_generator(hl_train, hl_validate, hl_test)
    
    return bb_train, bb_validate, bb_test

############## THE FOLLOWING 3 FUNCTIONS GENERATE FEATURES FOR X SETS, TARGET FEATURE NOT INCLUDED

def X_bed_to_bath_generator(X_train, X_validate, X_test):
    X_train["bath_to_bed_ratio"] = X_train.bath_rooms/X_train.bed_rooms
    X_validate["bath_to_bed_ratio"] = X_validate.bath_rooms/X_validate.bed_rooms
    X_test["bath_to_bed_ratio"] = X_test.bath_rooms/X_test.bed_rooms
    
    return X_train, X_validate, X_test

def X_house_to_lot_generator(X_train, X_validate, X_test):
    X_train["house_to_lot_sqft_ratio"]=X_train.house_square_feet/X_train.property_square_feet
    X_validate["house_to_lot_sqft_ratio"]=X_validate.house_square_feet/X_validate.property_square_feet
    X_test["house_to_lot_sqft_ratio"]=X_test.house_square_feet/X_test.property_square_feet
    
    return X_train, X_validate, X_test

def X_bath_to_house_generator(X_train, X_validate, X_test):
    X_train["bath_to_house_sqft_ratio"]=X_train.bath_rooms/X_train.house_square_feet
    X_validate["bath_to_house_sqft_ratio"]=X_validate.bath_rooms/X_validate.house_square_feet
    X_test["bath_to_house_sqft_ratio"]=X_test.bath_rooms/X_test.house_square_feet
    
    return X_train, X_validate, X_test


############## COMPILATION OF X FEATURE GENERATOR FUNCTIONS

def X_feature_generator(X_train, X_validate, X_test):
    X_bh_train, X_bh_validate, X_bh_test = X_bath_to_house_generator(X_train, X_validate, X_test)
    
    X_hl_train, X_hl_validate, X_hl_test = X_house_to_lot_generator(X_bh_train, X_bh_validate, X_bh_test)
    
    X_bb_train, X_bb_validate, X_bb_test = X_bed_to_bath_generator(X_hl_train, X_hl_validate, X_hl_test)
    
    return X_bb_train, X_bb_validate, X_bb_test
    
###################################################################################
                               # DATA VISUALIZATIONS #
###################################################################################


################# EXPLORATION VISUALS

##### HOUSE FEATURES

# first heat map for house value correlations
def hvalue_corr_heatmap(train):
    '''This function uses the train dataset to generate a heatmap showing the correlation of all features,
    except county, to property value'''
   
    # set figure dimensions
    fig = plt.figure(figsize=(8, 12))
    #make figure
    heatmap = sns.heatmap(train.drop(columns = ['county']).corr(method='spearman')[['property_value']].sort_values(by='property_value', ascending=False), vmin=-1, vmax=1, annot=True, cmap='YlGnBu')
    heatmap.set_title('Features Correlating with Home Price', fontdict={'fontsize':18}, pad=16);
    #summon figure
    plt.show


# heat map for house square foot correlation
def hsqft_corr_heatmap(train):
    '''This function uses the train dataset to generate a heatmap showing the correlation of all features,
    except county and property value, to house square feet'''
    
    #set figure dimensions
    fig = plt.figure(figsize=(8,12))
    #make figure
    heatmap=sns.heatmap(train.drop(columns = ['county', 'property_value']).corr(method='spearman')[['house_square_feet']].sort_values(by='house_square_feet', ascending=False), vmin=-1, vmax=1, annot=True, cmap='rocket')
    heatmap.set_title('Features Correlating with House Square Feet', fontdict={'fontsize':18}, pad=16);
    #summon figure
    plt.show()

# heat map for BTH correlation
def BTH_corr_heatmap(train):
    '''This function uses the train dataset to generate a heatmap showing the correlation of all features,
    except county and property value, to the engineered feature, bathroom to house ratio'''
    
    #set figure
    fig= plt.figure(figsize=(8, 12))
    #make figure
    heatmap = sns.heatmap(train.drop(columns = ['county', 'property_value']).corr(method='spearman')[['bath_to_house_sqft_ratio']].sort_values(by='bath_to_house_sqft_ratio', ascending=False), vmin=-1, vmax=1, annot=True, cmap='viridis')
    heatmap.set_title('Features Correlating with Bathroom to House Ratio', fontdict={'fontsize':18}, pad=16);
    #summon figure
    plt.show()
    
##### PROPERTY LOCATIONS

def correlation_by_county(train):

    sns.set()

    fig, axes = plt.subplots(2,2)#.figsize(12,8)

    sns.histplot(data =train, x = "year_built", hue = "county", ax=axes[0,0])
    sns.histplot(data=train, x= 'house_square_feet', hue = 'county', ax = axes[0,1])
    sns.histplot(data = train, x = "bath_to_bed_ratio", hue = "county", ax = axes[1,0])
    sns.histplot(data = train, x = "house_to_lot_sqft_ratio", hue = 'county', ax=axes[1,1])

    plt.show()
    
###################################################################################
                               # SCALING DATA #
###################################################################################

def scale_data(X_train, X_validate, X_test):
    
    # set scaler as Standard Scaler
    Stand_scaler = pre.StandardScaler()

    # Make copies of train, validate, and test dfs
    Stand_X_train_scaled = X_train.copy()
    Stand_X_validate_scaled = X_validate.copy()
    Stand_X_test_scaled = X_test.copy()

    # Fit scaler to Train df
    Stand_scaler.fit(X_train)

    # Transform data from three dfs
    Stand_X_train_scaled = Stand_scaler.transform(X_train)
    Stand_X_validate_scaled = Stand_scaler.transform(X_validate)
    Stand_X_test_scaled = Stand_scaler.transform(X_test)

    #Shove it back into a dataframe
    Stand_X_train_scaled = pd.DataFrame(Stand_X_train_scaled, columns = X_train.columns, index = X_train.index)
    Stand_X_validate_scaled = pd.DataFrame(Stand_X_validate_scaled, columns = X_validate.columns, index = X_validate.index)
    Stand_X_test_scaled = pd.DataFrame(Stand_X_test_scaled, columns = X_test.columns, index = X_test.index)


    return Stand_X_train_scaled, Stand_X_validate_scaled, Stand_X_test_scaled
    
    
    
def scaled_figure(X_train, X_train_scaled):
    # Plot figure
    plt.figure(figsize=(13, 6))
    plt.subplot(121)
    plt.hist(X_train, ec='black')
    plt.title('Original')
    plt.subplot(122)
    plt.hist(X_train_scaled, ec='black')
    plt.title('Scaled')    
    
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    




