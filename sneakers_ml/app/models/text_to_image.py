from pydantic import BaseModel


class Request(BaseModel):
    username: str
    text: str


class Response(BaseModel):
    class Metadata(BaseModel):
        title: str
        brand: str
        dataset: str
        price: str
        url: str

    images: list[str]
    metadata: list[Metadata]
