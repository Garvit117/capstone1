from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, lit
from pyspark.sql.window import Window
from pyspark.sql import functions as F

# Initialize Spark session with Hive support
spark = SparkSession.builder \
    .appName("Data Processing") \
    .enableHiveSupport() \
    .getOrCreate()

# Load data into Spark DataFrames
policy_df = spark.read.format("com.crealytics.spark.excel") \
    .option("useHeader", "true") \
    .load("Policy features.xlsx")

claim_df = spark.read.format("com.crealytics.spark.excel") \
    .option("useHeader", "true") \
    .load("Insurance claim.xlsx")

# Merge DataFrames
data_df = policy_df.join(claim_df, on="policy_id", how="inner")

# Handle duplicates
data_df = data_df.dropDuplicates()

# Handle missing values by filling with column mode
def fill_na_mode(df):
    for column in df.columns:
        mode = df.groupBy(column).count().orderBy("count", ascending=False).first()[0]
        df = df.withColumn(column, when(col(column).isNull(), mode).otherwise(col(column)))
    return df

new_df = fill_na_mode(data_df)

# Identify numeric and non-numeric columns
non_numeric = [item[0] for item in new_df.dtypes if item[1] == 'string']
numeric = [item[0] for item in new_df.dtypes if item[1] != 'string']
numeric.remove("is_claim")
non_numeric.remove("policy_id")

# Detect and handle outliers using IQR trimming
def iqr_trim(df, columns):
    for column in columns:
        quantiles = df.approxQuantile(column, [0.25, 0.75], 0.05)
        Q1 = quantiles[0]
        Q3 = quantiles[1]
        IQR = Q3 - Q1
        LB = Q1 - 1.5 * IQR
        UB = Q3 + 1.5 * IQR
        df = df.filter((col(column) >= LB) & (col(column) <= UB))
    return df

trimmed_df = iqr_trim(new_df, ["age_of_car", "age_of_policyholder", "population_density"])

# Save the processed DataFrame into Hive table
trimmed_df.write.mode("overwrite").saveAsTable("processed_data")

spark.stop()
