#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pandas import DataFrame
from joblib import load

BEST_MODEL_PATH = "./best_model.joblib" 

model = load(BEST_MODEL_PATH)

def inference(path)->list[int]:
    '''
    path: a DataFrame
    result is the output of function which should be 
    somethe like: [0,1,1,1,0]
    0 -> Lost
    1 -> Won
    '''
    result = []
    
    path.columns = ['customer', 'agent', 'sales_agent_emailID', 'contact_emailID', \
       'product', 'close_value', 'created_date', 'close_date']
    
    path.drop(['customer', 'agent','sales_agent_emailID', 'contact_emailID' ], axis=1, inplace=True)

    path = path.join(pd.get_dummies(path['product'])).drop('product', axis=1)
    
    path['diff_months'] = (path.close_date - path.created_date).dt.components.days/30.0
    
    path.drop(['created_date', 'close_date'], axis=1, inplace=True)
    
    
    result = model.predict(path)
    result = list(map(lambda label : 0 if label == 'Lost' else 1 , result))
    
    return result
    

