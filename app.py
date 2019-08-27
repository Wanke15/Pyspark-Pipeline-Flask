import json
import logging
import time

from flask import Flask
from flask import request
from pyspark.sql import SparkSession

from spark_util import load

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


def init():
    logger.info("Initializing app...")
    global sqlContext, model
    sqlContext = SparkSession.builder \
        .master("local") \
        .appName("MLPipeline predict") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    model = load("./models/test_model/")

    test = sqlContext.createDataFrame([
        (1, "hello")
        ], ["id", "text"])

    model.transform(test).collect()


@app.route('/spark/predict', methods=['GET'])
def predict():
    text = request.args.get('input')
    logger.warning(f'Get input: {text}')

    texts = [text for _ in range(1000)]

    test = sqlContext.createDataFrame([
        (idx, text)
        for idx, text in enumerate(texts)], ["id", "text"])
    # logger.warning(f"Created dataframe!")

    prediction = model.transform(test)
    # logger.warning("Model transform done!")

    selected = prediction.select("text", "prediction")
    # logger.warning("Select results done!")

    start = time.time()
    results = selected.collect()
    logger.info(f"Collect time consumed: {time.time() - start}")
    # logger.warning("Collect results done!")

    response = json.dumps([{"Input": row[0], "Class": row[1]} for row in results])
    # logger.warning("Parse results done!")

    return response


if __name__ == '__main__':
    init()
    app.run()
