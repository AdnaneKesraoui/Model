import great_expectations as ge
import pandas as pd


def clean_data(data, results):
    for result in results['results']:
        if not result['success']:
            expectation_type = result['expectation_config']['expectation_type']
            column = result['expectation_config']['kwargs']['column']

            if expectation_type == 'expect_column_values_to_not_be_null':
                # Fill missing values with a placeholder or the mean, if numeric
                if data[column].dtype == 'float64' or data[column].dtype == 'int64':
                    data[column].fillna(data[column].mean(), inplace=True)
                else:
                    data[column].fillna('UNKNOWN', inplace=True)
            
            elif expectation_type == 'expect_column_values_to_be_of_type':
                # Correct data types
                desired_type = result['expectation_config']['kwargs']['type_']
                try:
                    if desired_type == 'int64':
                        data[column] = data[column].astype('int64')
                    elif desired_type == 'float':
                        data[column] = data[column].astype('float')
                except Exception as e:
                    print(f"Error converting {column} to {desired_type}: {e}")

# Function to validate and revalidate tweets with cleaning
def validate_and_clean_tweets(tweets):
    tweets_ge = ge.dataset.PandasDataset(tweets)
    
    # Define expectations for tweets
    tweets_ge.expect_column_values_to_not_be_null("tweet")
    tweets_ge.expect_column_values_to_be_of_type("tweet", "object")
    tweets_ge.expect_column_values_to_match_regex("tweet", r".+")
    
    # Initial validation
    results = tweets_ge.validate()
    if not results['success']:
        clean_data(tweets, results)
        tweets_ge = ge.dataset.PandasDataset(tweets)  # Reload dataset for revalidation
        revalidation_results = tweets_ge.validate()
        print(revalidation_results)
    
    return tweets

# Function to validate and revalidate labels with cleaning
def validate_and_clean_labels(labels):
    labels_ge = ge.dataset.PandasDataset(labels)
    
    # Define expectations for labels
    labels_ge.expect_column_values_to_be_in_set("label", [0, 1, 2])
    labels_ge.expect_column_values_to_not_be_null("label")
    labels_ge.expect_column_values_to_be_of_type("label", "int64")
    
    # Initial validation
    results = labels_ge.validate()
    if not results['success']:
        clean_data(labels, results)
        labels_ge = ge.dataset.PandasDataset(labels)  # Reload dataset for revalidation
        revalidation_results = labels_ge.validate()
        print(revalidation_results)
    
    return labels

tweets = pd.read_csv('../test_text.txt', sep='\t', header=None, names=['tweet'])
labels = pd.read_csv('../test_labels.txt', sep='\t', header=None, names=['label'])

validated_tweets = validate_and_clean_tweets(tweets)
validated_labels = validate_and_clean_labels(labels)

validated_tweets.to_csv('validated_tweets.txt', sep='\t', header=True, index=False)
validated_labels.to_csv('validated_labels.txt', sep='\t', header=True, index=False)
