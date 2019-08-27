import json

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer

sqlContext = SparkSession.builder \
    .master("local") \
    .appName("MLPipeline example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
# Prepare training documents from a list of (id, text, label) tuples.
training = sqlContext.createDataFrame([
    (0, "a b c d e spark", 1.0),
    (1, "b d", 0.0),
    (2, "spark f g h", 1.0),
    (3, "hadoop mapreduce", 0.0)
], ["id", "text", "label"])

# Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
lr = LogisticRegression(maxIter=10, regParam=0.001)
pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])

# Fit the pipeline to training documents.
model = pipeline.fit(training)

# Prepare test documents, which are unlabeled (id, text) tuples.
test = sqlContext.createDataFrame([
    (4, "spark i j k"),
    (5, "l m n"),
    (6, "spark hadoop spark"),
    (7, "apache hadoop")
], ["id", "text"])

# Make predictions on test documents and print columns of interest.
prediction = model.transform(test)

model.save("./models/test_model")

import time

selected = prediction.select("id", "text", "probability", "prediction")
with open('./models/res.json', 'w') as f:
    start = time.time()
    pred_result = selected.collect()
    print("Collect time consumed: ", time.time() - start)
    json.dump([{"Input": row[1], "Class": row[3]} for row in pred_result], f)
print(json.dumps([{"Input": row[1], "Class": row[3]} for row in selected.collect()]))
# for row in selected.collect():
#     rid, text, prob, prediction = row
#     print("(%d, %s) --> prob=%s, prediction=%f" % (rid, text, str(prob), prediction))
