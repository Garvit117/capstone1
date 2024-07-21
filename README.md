from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, approxQuantile, expr
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize Spark session with Hive support
spark = SparkSession.builder \
    .appName("Data Processing") \
    .enableHiveSupport() \
    .getOrCreate()

# Read CSV files into Spark DataFrames
policy_df = spark.read.csv('Policy_features.csv', header=True, inferSchema=True)
claim_df = spark.read.csv('Insurance_claim.csv', header=True, inferSchema=True)

# Join DataFrames on policy_id
data_df = policy_df.join(claim_df, on="policy_id", how="inner")

# Handle duplicates
data_df = data_df.dropDuplicates()

# Handle missing values by filling with column mode
def fill_na_mode(df):
    for column in df.columns:
        mode = df.groupBy(column).agg(count('*').alias('count')).orderBy(col('count').desc()).first()[0]
        df = df.withColumn(column, when(col(column).isNull(), mode).otherwise(col(column)))
    return df

new_df = fill_na_mode(data_df)

# Identify numeric and non-numeric columns
non_numeric = [item[0] for item in new_df.dtypes if item[1] == 'string']
numeric = [item[0] for item in new_df.dtypes if item[1] != 'string']
numeric.remove("is_claim")
non_numeric.remove("policy_id")

# Detect columns with outliers using IQR
def detect_outliers(df, columns):
    outlier_columns = []
    bounds = {}
    for column in columns:
        quantiles = df.approxQuantile(column, [0.25, 0.75], 0.05)
        Q1 = quantiles[0]
        Q3 = quantiles[1]
        IQR = Q3 - Q1
        LB = Q1 - 1.5 * IQR
        UB = Q3 + 1.5 * IQR
        bounds[column] = (LB, UB)
        outliers = df.filter((col(column) < LB) | (col(column) > UB)).count()
        if outliers > 0:
            outlier_columns.append(column)
    return outlier_columns, bounds

# Detect columns with outliers and their bounds
outlier_columns, bounds = detect_outliers(new_df, numeric)

# Trim outliers using IQR
def iqr_trim(df, columns, bounds):
    for column in columns:
        LB, UB = bounds[column]
        df = df.filter((col(column) >= LB) & (col(column) <= UB))
    return df

trimmed_df = iqr_trim(new_df, outlier_columns, bounds)

# Collect data for plotting
numeric_data = {col: trimmed_df.select(col).rdd.flatMap(lambda x: x).collect() for col in numeric}

# Plot boxplots
fig, axs = plt.subplots(nrows=1, ncols=len(numeric), figsize=(7 * len(numeric), 4))
axs = axs.flatten()

for i, column in enumerate(numeric):
    sns.boxplot(x=numeric_data[column], ax=axs[i])
    axs[i].set_title(column)
    axs[i].set_ylabel("Value")

plt.tight_layout()
plt.show()

# Function to plot histograms
def hist_pltr(cdf, column):
    claim_df = cdf.filter(cdf['is_claim'] == 1).select(column).rdd.flatMap(lambda x: x).collect()
    no_claim_df = cdf.filter(cdf['is_claim'] == 0).select(column).rdd.flatMap(lambda x: x).collect()
    all_data = cdf.select(column).rdd.flatMap(lambda x: x).collect()

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    # All data
    axes[0].hist(all_data, bins=20, edgecolor='black')
    axes[0].set_title(f'{column} - All data')
    axes[0].set_xlabel(column)
    axes[0].set_ylabel('Frequency')

    # Data without claim
    axes[1].hist(no_claim_df, bins=20, edgecolor='black')
    axes[1].set_title(f'{column} - Without claim')
    axes[1].set_xlabel(column)
    axes[1].set_ylabel('Frequency')

    # Data with claim
    axes[2].hist(claim_df, bins=20, edgecolor='black')
    axes[2].set_title(f'{column} - With claim')
    axes[2].set_xlabel(column)
    axes[2].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

hist_pltr(trimmed_df, "policy_tenure")
hist_pltr(trimmed_df, "age_of_car")
hist_pltr(trimmed_df, "age_of_policyholder")
hist_pltr(trimmed_df, "population_density")

# Function to plot stacked histograms
def hist_pt(cdf, column):
    data = cdf.select(column, "is_claim").rdd.map(lambda row: (row[0], row[1])).collect()
    df = pd.DataFrame(data, columns=[column, "is_claim"])

    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x=column, hue="is_claim", multiple="stack", bins=20, edgecolor="black")
    plt.title(f'{column} - Classification on is_claim')
    plt.xlabel(column)
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

hist_pt(trimmed_df, "policy_tenure")
hist_pt(trimmed_df, "age_of_car")
hist_pt(trimmed_df, "age_of_policyholder")
hist_pt(trimmed_df, "population_density")

# Plot countplot
data = trimmed_df.select("area_cluster", "is_claim").rdd.map(lambda row: (row[0], row[1])).collect()
df = pd.DataFrame(data, columns=["area_cluster", "is_claim"])

fig, axs = plt.subplots(figsize=(16, 10))
sns.countplot(data=df, x="area_cluster", hue="is_claim")
plt.yscale("log")
plt.tick_params(labelrotation=45)
plt.tight_layout()
plt.show()

# Save the processed DataFrame into Hive table
trimmed_df.write.mode("overwrite").saveAsTable("processed_data")

# Stop the Spark session
spark.stop()
