import great_expectations as ge
import pandas as pd

# Load datasets
tweets = pd.read_csv('../test_text.txt', sep='\t', header=None, names=['tweet'])
labels = pd.read_csv('../test_labels.txt', sep='\t', header=None, names=['label'])

# Combine into a single DataFrame
data = pd.concat([tweets, labels], axis=1)

print(data.describe())

# Display information about data types and missing values
print(data.info())

# Convert to a Great Expectations dataset
data_ge = ge.dataset.PandasDataset(data)


data_ge.expect_column_values_to_be_in_set("label", [0, 1, 2])
data_ge.expect_column_values_to_not_be_null("tweet")
data_ge.expect_column_values_to_be_of_type("tweet", "object")
data_ge.expect_column_values_to_match_regex("tweet", r".+")
data_ge.expect_column_values_to_not_be_null("label")
data_ge.expect_column_values_to_be_of_type("label", "int")
data_ge.expect_column_values_to_match_regex("label", r".+")


data_ge.save_expectation_suite("data_expectations.json")

# Validate your dataset
results = data_ge.validate()
print(results)
