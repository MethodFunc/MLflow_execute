import mlflow


def set_experiments(name, artifact_uri, logger):
    #  Experiments 지정
    try:
        mlflow.create_experiment(name, artifact_location=artifact_uri)

    except mlflow.exceptions.MlflowException:
        logger.error(f'{name} is exists')

    mlflow.set_experiment(experiment_name=name)
