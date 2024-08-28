import os

from arize.experimental.datasets import ArizeDatasetsClient


ARIZE_DEVELOPER_KEY = os.getenv("ARIZE_DEVELOPER_KEY", None)
ARIZE_SPACE_ID = os.getenv("ARIZE_SPACE_ID", None)
ARIZE_DATASET_NAME = os.getenv("ARIZE_DATASET_NAME", "jailbreak_prompts")


def get_arize_datasets_client():
    return ArizeDatasetsClient(developer_key=ARIZE_DEVELOPER_KEY)
