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
	train = pd.read_csv('../train.csv', parse_dates=['datetime'])
	train['hour'] = pd.DatetimeIndex(train['datetime']).hour
        train['month'] = pd.DatetimeIndex(train['datetime']).month
	test = pd.read_csv('../test.csv', parse_dates=['datetime'])
	test['hour'] = pd.DatetimeIndex(test['datetime']).hour
	test['month'] = pd.DatetimeIndex(test['datetime']).month


	results = pd.DataFrame(columns=['datetime', 'count'])	
	for (month, hour), test_subset in test.groupby([test['month'], test['hour']]):
	    train_subset = train[(train['month'] == month) & (train['hour'] == hour)]
	    model = RandomForestRegressor(n_estimators=100)
	    model.fit(np.array(get_features(train_subset)), np.array(train_subset['count']))
	    predictions = model.predict(np.array(get_features(test_subset)))
	    dt = test_subset['datetime']
	    predictions = pd.Series(predictions, index=dt.index)
	    res = pd.concat([dt, predictions], axis=1)
	    res.columns=['datetime', 'count']
	    results = pd.concat([results, res])

	results['count'] = results['count'].astype('int64')
	results = results.sort('datetime')
	results.to_csv('../submissions/sixthSubmission.csv', index=False)


if __name__ == '__main__':
	main()
