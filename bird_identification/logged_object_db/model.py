from typing import Optional

from datetime import datetime

from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.engine import create_engine, Engine


# Model structure taken from the example at https://docs.sqlalchemy.org/en/20/orm/quickstart.html

class Base(DeclarativeBase):
    pass


class ClassLoggedObjectModel(Base):
    __tablename__ = "logged_object"

    id: Mapped[int] = mapped_column(primary_key=True)

    class_name: Mapped[Optional[str]]
    start_time: Mapped[datetime]
    end_time: Mapped[datetime]


def get_engine(url: str) -> Engine:
    engine = create_engine(url)

    Base.metadata.create_all(engine)

    return engine
