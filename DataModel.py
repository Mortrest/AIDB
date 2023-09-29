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
        


