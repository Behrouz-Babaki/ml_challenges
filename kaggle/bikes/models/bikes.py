#!/usr/bin/python

import csv
import time
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor

def main():
    train_sets = dict()
    values = dict()
    with open('train.csv') as t:
        tr=csv.DictReader(t)
        for d in tr:
            d2 = {k:d[k] for k in d if k!='count' and k!='datetime'}
            datetime = time.strptime(d['datetime'], '%Y-%m-%d %H:%M:%S')
            d2['hour'] = datetime.tm_hour
            k = datetime.tm_year, datetime.tm_mon 
            if k not in train_sets:
                train_sets[k] = []
                values[k] = []
            train_sets[k].append(d2)
            values[k].append(d['count'])

    vecs = dict()
    for k in train_sets:
        vecs[k] = DictVectorizer()
        train_sets[k] = vecs[k].fit_transform(train_sets[k])


    with open('test.csv') as s:
        print 'datetime,count'
        sr=csv.DictReader(s)
        models = dict()
        for d in sr:
            d2 = {k:d[k] for k in d if k!='datetime'}
            datetime = time.strptime(d['datetime'], '%Y-%m-%d %H:%M:%S')
            d2['hour'] = datetime.tm_hour
            k = datetime.tm_year, datetime.tm_mon 
            if k not in models:
                models[k] = RandomForestRegressor(n_estimators=100)
                models[k].fit(train_sets[k], values[k])
            res = models[k].predict(vecs[k].transform(d2))
            print '%s,%d' %(d['datetime'], res[0])
            
if __name__ == '__main__':
    main()
