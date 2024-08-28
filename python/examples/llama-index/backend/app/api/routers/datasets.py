from typing import Optional

from arize.experimental.datasets import ArizeDatasetsClient
from arize.experimental.datasets.utils.constants import GENERATIVE
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
import pandas as pd
from pydantic import BaseModel

from app.api.routers.utils import get_arize_datasets_client, ARIZE_SPACE_ID, ARIZE_DATASET_NAME


datasets_router = r = APIRouter()


class AddJailbreakPrompt(BaseModel):
    jailbreak_prompt: str


def get_dataset_id_by_name(client: ArizeDatasetsClient, dataset_name: str) -> Optional[str]:
    df = client.list_datasets(space_id=ARIZE_SPACE_ID)
    filtered_df = df.loc[df['dataset_name'] == dataset_name, 'dataset_id']
    if not filtered_df.empty:
        dataset_id = filtered_df.values[0]
    else:
        dataset_id = None
    return dataset_id


@r.post("/jailbreak_prompts")
async def add_jailbreak_prompts(data: AddJailbreakPrompt):
    try:
        client = get_arize_datasets_client()
        dataset_id = get_dataset_id_by_name(client, ARIZE_DATASET_NAME)
        if dataset_id:
            existing_dataset = client.get_dataset(space_id=ARIZE_SPACE_ID, dataset_id=dataset_id)
            old_jailbreak_prompts = existing_dataset[['jailbreak_prompt']].copy()
            new_jailbreak_prompt = pd.DataFrame([{'jailbreak_prompt': data.jailbreak_prompt}])
            df = pd.concat([old_jailbreak_prompts, new_jailbreak_prompt], ignore_index=True)
            dataset_id = client.update_dataset(space_id=ARIZE_SPACE_ID, dataset_id=dataset_id, data=df)
        else:
            df = pd.DataFrame([data.jailbreak_prompt], columns=['jailbreak_prompt'])
            dataset_id = client.create_dataset(space_id=ARIZE_SPACE_ID, dataset_name=ARIZE_DATASET_NAME, dataset_type=GENERATIVE, data=df)
        return {'dataset_id': dataset_id}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@r.get("/jailbreak_prompts")
async def get_jailbreak_prompts():
    try:
        client = get_arize_datasets_client()
        df = client.get_dataset(space_id=ARIZE_SPACE_ID, dataset_name=ARIZE_DATASET_NAME)
        data_dict = df.to_dict(orient='records')
        return JSONResponse(content=data_dict)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@r.delete("/jailbreak_prompts")
async def delete_jailbreak_prompts():
    try:
        client = get_arize_datasets_client()
        dataset_id = get_dataset_id_by_name(client, ARIZE_DATASET_NAME)
        if dataset_id:
            is_deleted = client.delete_dataset(space_id=ARIZE_SPACE_ID, dataset_id=dataset_id)
            return {'is_deleted': is_deleted}
        else:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Dataset Not Found.')
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
