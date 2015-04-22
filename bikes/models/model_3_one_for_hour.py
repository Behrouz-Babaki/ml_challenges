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
                 'windspeed']]

def update_df(data):
    dt = pd.DatetimeIndex(data['datetime'])
    data['hour'] = dt.hour
    data['month'] = dt.month
    data['weekday'] = dt.weekday


def main():
	train = pd.read_csv('../train.csv', parse_dates=['datetime'])
	test = pd.read_csv('../test.csv', parse_dates=['datetime'])
	update_df(train)
	update_df(test)

	results = pd.DataFrame(columns=['datetime', 'count'])
	for hour , test_subset in test.groupby('hour'):
	    train_subset = train[train['hour']==hour]
	    model = RandomForestRegressor(n_estimators=100)
	    model.fit(np.array(get_features(train_subset)), np.array(train_subset['count']))
	    predictions = np.array(model.predict(np.array(get_features(test_subset))))
	    dt = test_subset['datetime']
	    predictions = pd.Series(predictions, index=dt.index)
	    res = pd.concat([dt, predictions], axis=1)
	    res.columns=['datetime', 'count']
	    results = pd.concat([results, res])
	    
	results['count'] = results['count'].astype('int64')
	results = results.sort('datetime')

	results.to_csv('../submissions/fifthSubmission_.csv', index=False)

if __name__ == '__main__':
	main()
