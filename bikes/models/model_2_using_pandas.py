#!/usr/bin/python

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def get_features(data):
    return data[['season',
                 'holiday',
                 'workingday',
                 'weather',
                 'temp',
                 'atemp',
                 'humidity',
                 'windspeed',
                 'hour']]

def main():
	train = pd.read_csv('train.csv', parse_dates=['datetime'])
	train['hour'] = pd.DatetimeIndex(train['datetime']).hour
	test = pd.read_csv('test.csv', parse_dates=['datetime'])
	test['hour'] = pd.DatetimeIndex(test['datetime']).hour

	test_years = pd.DatetimeIndex(test['datetime']).year
	test_months = pd.DatetimeIndex(test['datetime']).month
	train_years = pd.DatetimeIndex(train['datetime']).year
	train_months = pd.DatetimeIndex(train['datetime']).month

        print 'datetime,count'
	for (year, month), test_subset in test.groupby([test_years, test_months]):
	    train_subset = train[(train_years == year) & (train_months == month)]
	    model = RandomForestRegressor(n_estimators=100)
	    model.fit(np.array(get_features(train_subset)), np.array(train_subset['count']))
	    predictions = model.predict(np.array(get_features(test_subset)))
	    dt = test_subset['datetime']
	    for i in range(len(predictions)):
		print '%s,%d' %(dt.ix[dt.index[i]],predictions[i])

if __name__ == '__main__':
	main()
