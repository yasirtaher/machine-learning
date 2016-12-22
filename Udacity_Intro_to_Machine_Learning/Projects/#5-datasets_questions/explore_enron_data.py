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
