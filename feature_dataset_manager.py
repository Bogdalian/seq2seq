import os
import pyspark.sql.functions as F
from pyspark.sql import SparkSession, Window
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType


class FeatureDatasetManager:
    def __init__(self, spark, actions_path, triggers_path, output_dir, force_recompute=False, min_actions=30, time_window=7):
        self.spark = spark
        self.actions_path = actions_path
        self.triggers_path = triggers_path
        self.output_dir = output_dir
        self.force_recompute = force_recompute
        self.min_actions = min_actions
        self.time_window = time_window
        os.makedirs(self.output_dir, exist_ok=True)  # Use exist_ok to avoid errors if dir exists

        self.actions_schema = StructType([
            StructField("guid", StringType(), True),
            StructField("date", TimestampType(), True),
            StructField("result", IntegerType(), True)
        ])

        self.triggers_schema = StructType([
            StructField("guid", StringType(), True),
            StructField("date", TimestampType(), True),
            StructField("trigger", IntegerType(), True),
            StructField("type", IntegerType(), True)
        ])

        self.filtered_df_schema = StructType([
            StructField("guid", StringType(), True),
            StructField("offer_date", TimestampType(), True),
            StructField("next_offer_date", TimestampType(), True),
            StructField("result", IntegerType(), True),
            StructField("action_date", TimestampType(), True),
            StructField("action", IntegerType(), True)
        ])

        self.filtered_df_path = os.path.join(self.output_dir, "filtered_df.parquet")
        self.classifier_data_path = os.path.join(self.output_dir, "classifier_data.parquet")

        self.offers = self._read_csv(actions_path, self.actions_schema)
        self.actions = self._read_csv(triggers_path, self.triggers_schema)

        if os.path.exists(self.filtered_df_path) and not self.force_recompute:
            self.filtered_df = self.load_filtered_df()
            print(f"Filtered_df loaded from {self.filtered_df_path}")
        else:
            self.offers = self._rename_and_filter_offers(self.offers)
            self.actions = self._rename_and_filter_actions(self.actions)
            self.filtered_df = self._filter_actions_before_offer()
            self._save_filtered_df()

        if os.path.exists(self.classifier_data_path) and not self.force_recompute:
            self.classifier_data = self.load_classifier_data()
            print(f"Classifier_data loaded from {self.classifier_data_path}")
        else:
            self.classifier_data = self.get_classifier_data()
            self._save_classifier_data()

    def _read_csv(self, path, schema):
        return self.spark.read.csv(path, header=True, schema=schema)

    def _rename_and_filter_offers(self, offers):
        windowSpec = Window.partitionBy("guid").orderBy("offer_date")
        return (offers
                .filter(F.col("date").isNotNull())
                .withColumnRenamed("date", "offer_date")
                .withColumn("next_offer_date", F.lead("offer_date").over(windowSpec))
               )

    def _rename_and_filter_actions(self, actions):
        return (actions
                .filter(F.col("date").isNotNull())
                .withColumnRenamed("date", "action_date")
                .withColumnRenamed("trigger", "action")
               )

    def _filter_actions_before_offer(self):
        joined_df = self.actions.join(self.offers, "guid", "inner")
        filtered_df = joined_df.filter(
            (F.col("action_date") < F.col("offer_date")) &
            (
                (F.col("next_offer_date").isNotNull() & (F.col("action_date") < F.col("next_offer_date"))) |
                (F.col("next_offer_date").isNull())
            )
        ).select("guid", "offer_date", "next_offer_date", "result", "action_date", "action")

        return filtered_df.orderBy("guid", "offer_date", "action_date")

    def get_embedding_data(self):
        return self.filtered_df

    def get_classifier_data(self):
        df = self.filtered_df
        df = df.filter(
            F.col("action_date") >= (F.col("offer_date") - F.expr(f"INTERVAL {self.time_window} DAYS"))
        )
        action_counts = df.groupBy("guid", "offer_date").count()
        qualified_groups = action_counts.filter(F.col("count") >= self.min_actions).select("guid", "offer_date")
        classifier_data_temp = df.join(qualified_groups, ["guid", "offer_date"], "inner")
        return classifier_data_temp

    def load_filtered_df(self):
        return self.spark.read.parquet(self.filtered_df_path, schema=self.filtered_df_schema)

    def _save_filtered_df(self):
        print(f"Saving filtered_df to {self.filtered_df_path}")
        self.filtered_df.write.mode("overwrite").parquet(self.filtered_df_path)

    def _save_classifier_data(self):
        print(f"Saving classifier_data to {self.classifier_data_path}")
        self.classifier_data.write.mode("overwrite").parquet(self.classifier_data_path)

    def load_classifier_data(self):
        return self.spark.read.parquet(self.classifier_data_path)


if __name__ == '__main__':
    spark = (
        SparkSession.builder
        .appName("Data Preparation")
        .master("local[*]")
        .config("spark.driver.memory", "2g")
        .config("spark.executor.memory", "2g")
        .config("spark.driver.maxResultSize", "1g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    actions_path = "seq2seq/test_data/actions.csv"
    triggers_path = "seq2seq/test_data/triggers.csv"
    output_dir = "output"

    data_manager = FeatureDatasetManager(spark,
                                         actions_path,
                                         triggers_path,
                                         output_dir,
                                         force_recompute=True,
                                         min_actions=20,
                                         time_window=7)

    spark.stop()
