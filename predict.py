import json
import logging
import time

from flask import Flask
from flask import request

from pyspark.sql import SparkSession

from spark_util import load

sqlContext = SparkSession.builder \
    .master("local") \
    .appName("MLPipeline predict") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

start = time.time()
model = load("./models/test_model/")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


logger.warning("Initializing app...")


@app.route('/spark/predict', methods=['GET'])
def predict():
    text = request.args.get('input')
    logger.warning(f'Get input: {text}')

    texts = [text for _ in range(1)]

    test = sqlContext.createDataFrame([
        (1, text)
        for text in texts], ["id", "text"])
    logger.warning(f"Created dataframe!")

    prediction = model.transform(test)
    logger.warning("Model transform done!")

    selected = prediction.select("text", "prediction")
    logger.warning("Select results done!")

    start = time.time()
    results = selected.collect()
    print("Collect time consumed: ", time.time() - start)
    logger.warning("Collect results done!")

    response = json.dumps([{"Input": row[0], "Class": row[1]} for row in results])
    logger.warning("Parse results done!")

    return response


if __name__ == '__main__':
    app.run()
