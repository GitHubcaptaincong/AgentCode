from typing import Optional, Any, Dict

from pydantic import Field, BaseModel


class Action(BaseModel):
    name: str = Field(description="Tool name")
    args: Optional[Dict[str, Any]] = Field(description="Tool input arguments, containing arguments names and values")