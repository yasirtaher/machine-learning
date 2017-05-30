#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print 'Number of people in the list {0}'.format(len(enron_data))

print 'Number of Feature for each {0}'.format(len(enron_data.values()[0]))

pois = [x for x, y in enron_data.items() if y['poi']]
print 'Number of POI\'s {0}'.format(len(pois))

poi_reader = open("../final_project/poi_names.txt","r")

poi_reader.readline() # skip url
poi_reader.readline() # skip blank line

poi_count = 0
for poi in poi_reader:
	poi_count += 1

print poi_count

#1 What is the total value of the stock belonging to James Prentice?
print enron_data["PRENTICE JAMES"]

# How many email messages do we have from Wesley Colwell to persons of interest?
print enron_data["COLWELL WESLEY"]

# Whats the value of stock options exercised by Jeffrey K Skilling?
print enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]

# Follow the Money
names = ['SKILLING JEFFREY K', 'FASTOW ANDREW S', 'LAY KENNETH L']
names_payments = {name:enron_data[name]['total_payments'] for name in names}
print sorted(names_payments.items(), key=lambda x: x[1], reverse=True)
