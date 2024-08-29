# created following instructions from this tutorial https://www.youtube.com/watch?v=U9ReNx8E61g
import pandas as pd
from upgini import FeaturesEnricher, SearchKey
from upgini.metadata import CVType
from catboost import CatBoostRegressor
from catboost.utils import eval_metric

#take data from the CSV file
df_path = "train.csv.zip"
df = pd.read_csv(df_path)
df =df.sample(n=19_000, random_state=0)

#convert data to strings
df["store"] = df["store"].astype(str)
df["item"] = df["item"].astype(str)

df["date"] = pd.to_datetime(df["date"])

#sort data by date
df.sort_values(by="date", inplace=True)
df.reset_index(inplace=True, drop=True)

#set training and testing data, training data is before 2017, testing data is after 2017
train = df[df["date"] < "2017-01-01"]
test = df[df["date"] >= "2017-01-01"]

#deisgnate features and labels
train_features = train.drop(columns="sales")
train_target = train["sales"]
test_features = test.drop(columns="sales")
test_target = test["sales"]


#enrich features, i.e bring in new data for the sake of accuracy

enricher = FeaturesEnricher(search_keys={
        "date":SearchKey.DATE #search for the date data
        },
        cv = CVType.time_series
)

enricher.fit(train_features, train_target, eval_set=[(test_features, test_target)])


#implement the catboost algorithm

model = CatBoostRegressor(verbose=False, allow_writing_files=False, random_state=0)

enricher.calculate_metrics(
    train_features, train_target, eval_set=[(test_features, test_target)], estimator=model, scoring="mean_absolute_error"
)

#train the model
enriched_train_features = enricher.transform(train_features, keepinput=True)
enriched_test_features = enricher.transform(test_features, keepinput=True)

#test the model without the enriched data from upgini
model.fit(train_features, train_target)
predictions = model.predict(test_features)
eval_metric(test_target.values, predictions, "SMAPE")

#test the model with the enriched data from upgini
model.fit(enriched_train_features, train_target)
enriched_predictions = model.predict(enriched_test_features)
eval_metric(test_target.values, enriched_predictions, "SMAPE")