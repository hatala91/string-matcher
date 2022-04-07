import pandas as pd
from azureml.core import Dataset, Run, Workspace

def _get_workspace() -> Workspace:
    try:
        run = Run.get_context()
        return run.experiment.workspace
    except Exception:
        from .config import (
            RESOURCE_GROUP,
            SUBSCRIPTION_ID,
            WORKSPACE_NAME
        )

        return Workspace(SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME)

def get_dataset_as_dataframe(name: str) -> pd.DataFrame:
    workspace = _get_workspace()
    dataset = Dataset.get_by_name(workspace, name)
    return dataset.to_pandas_dataframe()
