# import great_expectations as ge
# import pandas as pd

# # Load datasets
# tweets = pd.read_csv('../test_text.txt', sep='\t', header=None, names=['tweet'])
# labels = pd.read_csv('../test_labels.txt', sep='\t', header=None, names=['label'])

# # Combine into a single DataFrame
# data = pd.concat([tweets, labels], axis=1)

# print(data.describe())

# # Display information about data types and missing values
# print(data.info())

# # Convert to a Great Expectations dataset
# data_ge = ge.dataset.PandasDataset(data)


# data_ge.expect_column_values_to_be_in_set("label", [0, 1, 2])
# data_ge.expect_column_values_to_not_be_null("tweet")
# data_ge.expect_column_values_to_be_of_type("tweet", "object")
# data_ge.expect_column_values_to_match_regex("tweet", r".+")
# data_ge.expect_column_values_to_not_be_null("label")
# data_ge.expect_column_values_to_be_of_type("label", "int")


# data_ge.save_expectation_suite("data_expectations.json")

# # Validate your dataset
# results = data_ge.validate()
# print(results)

# data_cleaned = data[data['label'].isin([0, 1, 2])]
import great_expectations as ge
import pandas as pd

# Load datasets
tweets = pd.read_csv('../test_text.txt', sep='\t', header=None, names=['tweet'])
labels = pd.read_csv('../test_labels.txt', sep='\t', header=None, names=['label'])

# Combine into a single DataFrame
data = pd.concat([tweets, labels], axis=1)


# Continue with data validation and cleaning as before
data_ge = ge.dataset.PandasDataset(data)

# Define expectations
data_ge.expect_column_values_to_be_in_set("label", [0, 1, 2])
data_ge.expect_column_values_to_not_be_null("tweet")
data_ge.expect_column_values_to_be_of_type("tweet", "object")
data_ge.expect_column_values_to_match_regex("tweet", r".+")
data_ge.expect_column_values_to_not_be_null("label")
data_ge.expect_column_values_to_be_of_type("label", "int64")

# Save expectation suite
data_ge.save_expectation_suite("data_expectations.json")

# Validate your dataset
results = data_ge.validate()
print(results)

if not results['success']:
    for result in results['results']:
        if not result['success']:
            expectation_type = result['expectation_config']['expectation_type']
            
            # Example: Handle missing values
            if expectation_type == 'expect_column_values_to_not_be_null':
                column = result['expectation_config']['kwargs']['column']
                # Example strategy: Fill missing values with a placeholder or the mean, if numeric
                if data[column].dtype == 'float64' or data[column].dtype == 'int64':
                    data[column].fillna(data[column].mean(), inplace=True)
                else:
                    data[column].fillna('UNKNOWN', inplace=True)
            
            # Example: Correct data types
            elif expectation_type == 'expect_column_values_to_be_of_type':
                column = result['expectation_config']['kwargs']['column']
                desired_type = result['expectation_config']['kwargs']['type_']
                # Attempt to convert column to the desired type
                try:
                    if desired_type == 'int64':
                        data[column] = data[column].astype('int64')
                    elif desired_type == 'float':
                        data[column] = data[column].astype('float')
                    # Add more type conversions as needed
                except Exception as e:
                    print(f"Error converting {column} to {desired_type}: {e}")
            
            # Add more corrective actions based on other types of expectations as needed

# After corrections, you might want to validate again to check if issues are resolved
revalidation_results = data_ge.validate()
print(revalidation_results)



