"""Named parameters: the values a design is driven by. A parameter's `expr` is a number ("120") or a formula of other
parameters ("height * 0.8"); a text parameter's `expr` is its text, never evaluated."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator

from .expr import valid_name

Quantity = Literal["length", "angle", "ratio", "count", "bool", "text"]


class Parameter(BaseModel):
    name: str
    quantity: Quantity = "length"
    expr: str
    comment: str = ""

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not valid_name(v):
            raise ValueError(f"{v!r} cannot be a parameter name: use letters, digits and _, not starting with a digit, "
                             "and not a function or unit name")
        return v
