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
                 'hour', 
                 'weekday',
                 'isweekend',
                 'avg'
                 ]]

def main():
	train = pd.read_csv('../train.csv', parse_dates=['datetime'])
	train['hour'] = pd.DatetimeIndex(train['datetime']).hour
	train['year'] = pd.DatetimeIndex(train['datetime']).year
	train['month'] = pd.DatetimeIndex(train['datetime']).month
	train['weekday'] = pd.DatetimeIndex(train['datetime']).weekday
        train['isweekend'] = 0
        train.loc[(train['weekday']==5) | (train['weekday']==6), 'isweekend'] = 1

	test = pd.read_csv('../test.csv', parse_dates=['datetime'])
	test['hour'] = pd.DatetimeIndex(test['datetime']).hour
	test['year'] = pd.DatetimeIndex(test['datetime']).year
	test['month'] = pd.DatetimeIndex(test['datetime']).month
	test['weekday'] = pd.DatetimeIndex(test['datetime']).weekday
        test['isweekend'] = 0
        test.loc[(test['weekday']==5) | (test['weekday']==6), 'isweekend'] = 1

        averages = train.groupby([train['year'], train['month']])['count'].mean()
        for (year, month), group in test.groupby([test['year'], test['month']]):
            train['avg'] = averages.loc[year, month]
            test['avg'] = averages.loc[year, month]

	results = pd.DataFrame(columns=['datetime', 'count'])	
	for (hour,iw),  test_subset in test.groupby([test['hour'], test['isweekend']]):
	    train_subset = train[(train['hour'] == hour)& (train['isweekend'] == iw)]
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
	results.to_csv('../submissions/eighthSubmission.csv', index=False)


if __name__ == '__main__':
	main()
