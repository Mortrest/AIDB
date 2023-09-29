# AIDB
For this project, I used [this](https://www.kaggle.com/datasets/sshikamaru/car-object-detection) unstructured dataset consisting of frames of car images. I employed a pre-trained YoloV8 model for object detection, which outputs the x and y coordinates, as well as the confidence of each car detected in each picture. The architecture of the project consists of two main classes.

The first class is the DataModel, which has three main components. The first component is the base table, consisting of the IDs of each image as a primary key. The second component is the mappings, which is a dictionary (although it should have been a class) with four attributes: the table name, the prediction function, the input columns, and the output columns. The third component is the user-defined metadata, which I did not implement as I find it irrelevant.

The second class is the AIDB class, which includes the execute query and approximate query methods. It also has optimization and caching functions. The project works as follows:

1. The user needs to define the mappings for each model. Based on these attributes, tables with t d, with null values for each entry.
2. With the null entries, virtual tables are defined. Each query can be fully executed using SQL via DuckDB. The resulting table is then materialized, after filtering and other operations.
3. Before materializing each table, the system checks whether it has been cached in the ML Model table or not. If it hasn't, the uncached data is sent to the predictor, and then it is aggressively cached.
4. Lastly, the system has the ability to answer approximate queries using stratified sampling.
