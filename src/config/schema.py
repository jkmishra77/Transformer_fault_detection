from pydantic import BaseModel, Field

class TrainConfig(BaseModel):
    model_type: str
    model_params: dict
    random_state: int
    test_size: float
    target_column: str
