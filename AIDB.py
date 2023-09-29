
import duckdb 
import pandas as pd
from DataModel import DataModel
from ml_model import predict
import copy
"""
Notes: 
- Column names for each model should be unique
"""

__author__ = "ALI FALAHATI"


def generate_cols_ml1(df, data_model, table_name, ml_model) -> pd.DataFrame:
    """ Calls the predict function from ml_model.py """
    model_df = pd.read_csv("models_table.csv")

    # TODO: Make it compatible with mappings
    image_ids = df['id'].tolist()
    res = {}
    true_images_ids = []
    for i in image_ids:
        ref = model_df.loc[model_df['id'] == i][table_name]
        if ref.any() == 1:
            res[i] = eval(model_df.loc[model_df['id'] == i][table_name + "_data"].values[0])
        else: 
            true_images_ids.append(i)

    if len(true_images_ids) > 0:
        pred = predict(true_images_ids)

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
    # Caching
    model_df.to_csv("models_table.csv", index= False)
    output = {}

    # Cols extracted from the dataModel mappings of the corresponding model
    cols = copy.copy(data_model.mappings[ml_model]['output'])
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



class AIDB:
    def __init__(self, DataModel, db='file2.db') -> None:
        self.data_model = DataModel
        self.db = db
        self.cols_mapping = {}
        self.table = self._create_virtual_tables()
        self.models_table = self._create_models_tables()

    def _create_models_tables(self) -> pd.DataFrame:
        """ Creating models table for caching """
        data_model = self.data_model
        dm = data_model.base_table

        for m in data_model.mappings.keys():
            dm[data_model.mappings[m]['table']] = 0
            dm[data_model.mappings[m]['table'] + "_data"] = None
        dm.to_csv("models_table.csv")

        return dm

    def _add_col_mapping(self, lst, model_name) -> None:
        """" Creating a one-to-one mapping for each column of the each table to the corresponding model """
        for l in lst:
            self.cols_mapping[l] = model_name

    def _create_virtual_tables(self) -> list:
        """ Crating the virtual tables neccesary for querying """
        data_model = self.data_model
        con = duckdb.connect(self.db)
        lst = []
        df = data_model.base_table
        try:
            con.sql(f"CREATE TABLE base_table AS SELECT * FROM df")
        except:
            pass
        mappings = data_model.mappings

        for m in mappings.keys():
            table_name = mappings[m]['table']
            # Base table - parent
            sample_df = data_model.base_table[mappings[m]['input'][0]['cols']]
            if len(mappings[m]['input']) > 1:
                for i in mappings[m]['input']:
                    if i['table'] != 'base_table':
                        sample_df[i['cols']] = None
            sample_df[mappings[m]['output']] = None
            self._add_col_mapping(mappings[m]['output'], m)
            lst.append(sample_df)
            try:
                con.sql(f"CREATE TABLE {table_name} AS SELECT * FROM sample_df")
            except:
                pass

        return lst
    
    def _check_for_condition(self, query, con) -> bool:
        """ Checks whether the query condition needs any information from generated rows """
        query = query.lower()
        query_pre_where = query.split("where")[-1].split("limit")[0].split(" ")
        required_models = []
        for q in query_pre_where:
            if q in self.cols_mapping.keys():
                required_models.append(self.cols_mapping[q])
        required_models = set(required_models)
        if len(required_models) > 0:
            """ It means that we have first materialize the rows first """
            for m in required_models:
                table_name = 'cars'
                model_name = 'ml1'
                output = self.data_model.mappings['ml1']['function'](self.data_model.base_table, self.data_model, table_name, model_name)
                try:
                    con.sql("DROP TABLE temp")
                except:
                    pass
                con.sql("CREATE TABLE temp AS SELECT * FROM output")
                query = query.replace("cars", "temp")
                df = con.sql(query).df()
                print(df)
                return False

        return True
    
    def stratified_sampling(self, mode, target):
        df = self.data_model.base_table.sample(50, replace=False)
        table_name = 'cars'
        model_name = 'ml1'
        output = dm.mappings['ml1']['function'](df, self.data_model, table_name, model_name)

        first = output.dropna()
        first['cnt'] = 1
        second = first.groupby("id").sum()
        second['cnt_2'] = 1
        output = output.dropna()
        ret = 0
        for x,y in output.iterrows():
            output.loc[x, 'cnt'] = int(second[second.index == y['id']]['cnt'].values[0])

        strata_size = len(output) // len(output['cnt'].unique())
        # Create an empty list to store the sampled data
        sampled_data = []
        # Iterate over each stratum
        for i in output['cnt'].unique():
            # Get a random sample from each stratum
            stratum = output[output['cnt'] == i].sample(strata_size, replace = True)[target]
            # Add the sample to the sampled_data list
            sampled_data.extend(stratum)

        if mode == 'AVG':
            ret = sum(sampled_data) / len(sampled_data)
        elif mode == 'MIN':
            ret = min(sampled_data)
        elif mode == 'MAX':
            ret = max(sampled_data)
        return ret

    def _find_mode(self, query):
        lst = ["AVG", "MIN", "MAX"]
        target, mode = None, None
        for l in lst:
            if l in query:
                target = query.split(f"{l}(")[-1].split(")")[0]
                mode = l
        return target, mode
    def _estimate(self, query):
        if "AVG" in query or "MIN" in query or "MAX" in query:
            return True
        return False

    def execute_query(self, query, approx = True):
        """ Execute the shallow query and then generate the required rows """
        con = duckdb.connect(self.db)
        if approx and self._estimate(query):
            target, mode = self._find_mode(query)
            print(self.stratified_sampling(mode, target))
        elif self._check_for_condition(query, con):
            # TODO: if the query contain AVG, DIST it would be wrong so u have to change it beforehand
            df = con.sql(query).df()
            # Works only for one query but easily could be generalized
            table_name = 'cars'
            model_name = 'ml1'
            output = self.data_model.mappings['ml1']['function'](df, self.data_model, table_name, model_name)
            return output



