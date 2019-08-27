import json
import time

from pyspark.sql import SparkSession

from spark_util import load

sqlContext = SparkSession.builder \
    .master("local") \
    .appName("MLPipeline predict") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

model = load("./models/test_model/")

test = sqlContext.createDataFrame([
    (4, "spark i j k"),
    (5, "l m n"),
    (6, "spark hadoop spark"),
    (7, "apache hadoop")
], ["id", "text"])

# Make predictions on test documents and print columns of interest.
prediction = model.transform(test)

selected = prediction.select("id", "text", "prediction")
pred_result = selected.collect()
print(json.dumps([{"Id": row[0], "Input": row[1], "Class": row[2]} for row in pred_result]))
