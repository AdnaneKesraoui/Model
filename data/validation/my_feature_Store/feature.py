from datetime import timedelta

from feast import Entity, Feature, FeatureView, FileSource, ValueType

tweet_source = FileSource(
    path="../../val_text.txt",
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created_timestamp",
)

tweet_entity = Entity(name="tweet_id", value_type=ValueType.INT64, description="Tweet ID")

tweet_feature_view = FeatureView(
    name="tweet_preprocessed_features",
    entities=["tweet_id"],
    ttl=timedelta(days=1),
    features=[
        Feature(name="preprocessed_text", dtype=ValueType.STRING),
    ],
    batch_source=tweet_source,
    tags={},
)

