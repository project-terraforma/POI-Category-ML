
import json
# import pandas as pd
# from tabulate import tabulate # not needed, used to print tables

def getDB():
    # open csv file and add all contents to variable
    pathToFile = "./data/yelp_academic_dataset_business.json"
    db = []
    with open(pathToFile, "r", encoding="utf-8") as db_file:
        for line in db_file:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            db.append(obj)
    
    return db

def getHeaders():
    # returns a list of all headers of the json list
    db = getDB()
    headers = list(db[0].keys())
    return headers


def getCategories(db):
    # return all categories of businesses from db
    categories = []
    head = getHeaders() 
    for business in db:
        curr = business[head[12]] 
        categories.append(business[head[12]])
    return categories

def writeToFile(db):
    # add unique businesses
    with open("yelp_categories.txt", "a") as f:
        for item in db:        
            f.write(str(item) + ", \n")
        return (len(db))

def get_unique_categories():
        
    db_json = getDB()
    cat_data = getCategories(db_json)

    eachCat = []
    for category in cat_data:
        if (category not in eachCat):
            eachCat.append(category)

    unique = set()

    def process(entry):
        if not isinstance(entry, str):
            return
        for cat in entry.split(','):
            cat = cat.strip()
            if cat:
                unique.add(cat)
    
    # Normalize input
    if isinstance(cat_data, str) or cat_data is None:
        process(cat_data)
    elif isinstance(cat_data, list):
        for item in cat_data:
            # Recursively process sub-lists if necessary
            if isinstance(item, list):
                for sub in item:
                    process(sub)
            else:
                process(item)
    return unique

def get_input_data():
    db = getDB()
    train_db = {}
    head = getHeaders()
    
    # Ensure the indices are correct
    category_key = head[12]  # Replace with the correct header name if needed
    name_key = head[1]       # Replace with the correct header name if needed
    
    for m in db:
        curr_cat = m.get(category_key, None)  # Safely get category
        curr_name = m.get(name_key, None)    # Safely get business name
        
        if curr_name and curr_cat:  # Ensure both name and category exist
            train_db[curr_name] = curr_cat

    return train_db

# LLM Generated Category Matches
# ### Here are the summary statistics of the partial-match mapping:
# #### Total Overture leaf categories: 2,116
# ##### Overture categories with ≥1 Yelp match: 1,191
# 
# ##### Unmatched Overture categories (outliers): 925
# 
# ##### Total Overture–Yelp match pairs: 1,531
# 
# ##### Unique Yelp categories matched: 1,084
'''
db_json = getDB()
print(f"Loaded {len(db_json)} business records")

# add all categories of all businesses
categories = getCategories(db_json)

eachCat = []
for category in categories:
    if (category not in eachCat):
        eachCat.append(category)
        # f.write(str(category) + ", \n") 


unq_categories = get_unique_categories(eachCat)

# print(len(eachCat)) 83161 different categories overall
# print(unq_categories)

# fileWrite = writeToFile(unq_categories)
# print(f"Lines written: {fileWrite}")



print(getHeaders())

# Here I want to define that returns a dictionary with the business name as key and the categories as values


# train = get_input_data() # this function works
# print(train["Starbucks"])

'''

