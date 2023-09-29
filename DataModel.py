"""
Code for QueryProcessor class and functions to assist.
"""

import pandas as pd
from ml_model import predict
from pathlib import Path
import os


__author__ = "ALI FALAHATI"



class DataModel:
    def __init__(self, data_path, images_path, mappings, metadata = {}) -> None:
        self.data_path = data_path
        self.images_path = images_path
        self.mappings = mappings
        self.metadata = metadata
        self.base_table = self.find_or_create_base(data_path, images_path)
        self.tables = {
            'base_table': self.base_table
            }

    def find_or_create_base(self, data_path, images_path) -> pd.DataFrame:
        """ Check if there's a table, otherwise create it """
        data_path = Path(data_path)
        if not data_path.exists():
            images_path = Path(images_path)
            file_names = os.listdir(images_path)
            df = pd.DataFrame({'id' : file_names})
            df.to_csv("data.csv", index= False)
        return pd.read_csv("data.csv")

    def add_virtualtable(self, ml_model, df) -> None:

        self.tables[ml_model] = df


    def set_mappings(self, ml_model, cols) -> None:
        
        self.mappings[ml_model] = cols
        """
        mappings would be like:
        {
            ml1: {
                'table' : 'cars',
                'function' : generate_cols_ml1,
                'input': {
                    'table' : [cols]
                },
                'output: ['id', 'x', 'y', 'confidence']
            }
            ....
        }
        """

    def set_metadata(self, ml_model, dataframe) -> None:

        self.metadata[ml_model] = dataframe
        """
        metadata would be like:
        {
            ml1: 
            {
                'timestamp' : {'id1' : 23, 'id2' : 12, ...},
                ...
            }, 
            ....
        }
        """
        


def generate_cols_ml1(df, data_model, table_name, ml_model) -> pd.DataFrame:
    """ Calls the predict function from ml_model.py """
    # query = []
    # df = pd.read_csv("data.csv")
    model_df = pd.read_csv("models_table.csv")

    # TODO: Make it compatible with mappings
    image_ids = df['id'].tolist()
    res = {}

    for i in image_ids:
        ref = model_df.loc[model_df['id'] == i][table_name]
        if ref.any() == 1:
            res[i] = eval(model_df.loc[model_df['id'] == i][table_name + "_data"].values[0])
            image_ids.remove(i)

    if len(image_ids) > 0:
        pred = predict(image_ids)

        """ Caching the results from the model"""
        for p in pred:
            item = res.get(p['id'])
            if item != None:
                res[p['id']].append(p)
            else:
                res[p['id']] = [p]

        for d in res.keys():
            model_df.loc[model_df['id'] == d, table_name + "_data"] = str(res[d])
            model_df.loc[model_df['id'] == d, table_name] = 1

    """ res would be a dictionary like
  {'vid_4_1000.jpg': [{'id': 'vid_4_1000.jpg',
   'x': 290.83704,
   'y': 194.49673,
   'conf': 0.6697502136230469},
  {'id': 'vid_4_1000.jpg',
   'x': 246.29236,
   'y': 206.23581,
   'conf': 0.27807939052581787}],...
   """

    # Caching
    model_df.to_csv("models_table.csv", index= False)
    output = {}

    # Cols extracted from the dataModel mappings of the corresponding model
    cols = data_model.mappings[ml_model]['output']
    cols.insert(0, 'id')
    # initializing
    for col in cols:
        output[col] = []
    
    for value in res.values():
        for val in value:
            for col in cols:
                output[col].append(val[col])
    dataframe = pd.DataFrame(output)
    dataframe.dropna(how='all', inplace=True)
    data_model.add_virtualtable(ml_model, dataframe)
    return dataframe
