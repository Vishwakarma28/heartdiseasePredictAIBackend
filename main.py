import pandas as pd
import os
import joblib 
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


MODEL_FILE = 'model.pkl'
PIPELINE_FILE = 'pipeline.pkl'

def build_pipeline(num_attri,cat_attri):

    num_pipeline = Pipeline([
        ('imputer',SimpleImputer(strategy='median')),
        ('scaler',StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('Onehot',OneHotEncoder(handle_unknown='ignore'))
    ])

    full_pipeline  = ColumnTransformer([
        ('num',num_pipeline,num_attri),
        ('cat',cat_pipeline,cat_attri)
    ])

    return full_pipeline

if not os.path.exists(MODEL_FILE):
    df = pd.read_csv('heart2.csv')

    split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=40)

    for train_index , test_index in split.split(df,df['Sex']):
        df.loc[test_index].to_csv('test.csv',index=False)
        strat_train_set = df.loc[train_index]

    feature_data = strat_train_set.drop('HeartDisease',axis=1) 
    labels_data = strat_train_set['HeartDisease']   

    # print(feature_data)
    # print(labels_data)   
    num_attri = strat_train_set.drop(['Sex','ChestPainType','RestingECG','ExerciseAngina','Oldpeak','ST_Slope','HeartDisease'],axis=1).columns.tolist()
    cat_attri = ['Sex','ChestPainType','RestingECG','ExerciseAngina','Oldpeak','ST_Slope']
    # print("num",num_attri)
    # print('cat',cat_attri)  


    pipeline = build_pipeline(num_attri,cat_attri)
    prepared_data = pipeline.fit_transform(feature_data)

    # print(pipeline)
    # print(prepared_data)
    
    # model = RandomForestRegressor(random_state=40)
    model = DecisionTreeRegressor(random_state=40)
    

    model.fit(prepared_data,labels_data)

    joblib.dump(model,MODEL_FILE)
    joblib.dump(pipeline,PIPELINE_FILE)

else:
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv('test.csv')

    transformed_input = pipeline.transform(input_data)
    pridict = model.predict(transformed_input)
    input_data['HeartDisease'] = pridict
    input_data.to_csv("output.csv" , index=False)
    print("infreance done")