from pyspark.ml import PipelineModel


def load(model_path):
    model = PipelineModel.load(model_path)
    return model
