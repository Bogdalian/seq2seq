import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from ptls.preprocessing import PysparkDataPreprocessor

class ModelDataPreparator:
    def __init__(self, spark, output_dir):
        self.spark = spark
        self.output_dir = output_dir

    def read_data(self, data_path):
        """
        Reads a parquet file into a Spark DataFrame.
        """
        df = self.spark.read.parquet(data_path)
        return df

    def preprocess_data(self, df, col_id):
        """
        Preprocesses the input DataFrame using PysparkDataPreprocessor.
        """
        preprocessor = PysparkDataPreprocessor(
            col_id=col_id,
            col_event_time='action_date',
            event_time_transformation='dt_to_timestamp',
            cols_category=['action']
        )
        preprocessed_df = preprocessor.fit_transform(df)
        return preprocessed_df

    def save_data(self, df, output_path):
        """
        Saves the DataFrame to the specified output path.
        """
        print(f"Saving data to {output_path}")
        df.write.mode("overwrite").parquet(output_path)

    def process_and_save(self, input_path, output_path, col_id, data_name="data"):
        """
        Reads data from input_path, preprocesses it, and saves it to output_path.
        """
        print(f"Processing {data_name} from {input_path}...")
        df = self.read_data(input_path)
        preprocessed_data = self.preprocess_data(df, col_id).select(col_id, "event_time", 'action', 'result')
        self.save_data(preprocessed_data, output_path)
        preprocessed_data.printSchema()
        return preprocessed_data


if __name__ == '__main__':
    spark = SparkSession.builder.appName("ModelDataPrep") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.maxResultSize", "2g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    
 
    filtered_df_path = 'output/filtered_df.parquet'
    classifier_data_path = 'output/classifier_data.parquet'
    output_dir = 'output'
    preprocessed_filtered_df_path = f'{output_dir}/preprocessed_filtered_df.parquet'
    preprocessed_classifier_data_path = f'{output_dir}/preprocessed_classifier_data.parquet'


    data_preparator = ModelDataPreparator(spark, output_dir)

 
    data_preparator.process_and_save(input_path=filtered_df_path,
                                     output_path=preprocessed_filtered_df_path,
                                     col_id='guid',
                                     data_name="preprocessed_filtered_df")
    data_preparator.process_and_save(classifier_data_path, 
                                     preprocessed_classifier_data_path,
                                     col_id='offer_date',
                                     data_name="preprocessed_classifier_data")

    spark.stop()
