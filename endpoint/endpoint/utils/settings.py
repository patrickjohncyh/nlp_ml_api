from os import getenv
from pydantic import BaseSettings


class Settings(BaseSettings):
    app_name: str = "endpoint"
    mode: str
    dbpath: str

    class Config:
        env_file = f"endpoint/envs/{getenv('MODE')}.env"
