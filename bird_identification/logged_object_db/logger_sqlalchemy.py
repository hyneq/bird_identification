from dataclasses import dataclass

from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from ..tracking.logger import ListObjectLogger, IObjectLoggerFactory, ClassLoggedObject

from .model import ClassLoggedObjectModel, get_engine

def get_orm(o: ClassLoggedObject):
    return ClassLoggedObjectModel(
        class_name=o.class_name,
        start_time=o.start_time,
        end_time=o.end_time
    )


class SQLAlchemyClassLogger(ListObjectLogger[ClassLoggedObject]):

    engine: Engine

    def __init__(self, engine: Engine):
        super().__init__()

        self.engine = engine
        self.Session = sessionmaker(engine)


    def _log(self):
        with self.Session.begin() as session:
            for o in self.objects:
                session.add(get_orm(o))


@dataclass
class SQLAlchemyClassLoggerFactory(IObjectLoggerFactory[ClassLoggedObject]):

    name = "sqlalchemy"

    def __call__(self,
        path: str,
        *_, **__
    ):
        engine = get_engine(path)

        return SQLAlchemyClassLogger(engine)


factory = SQLAlchemyClassLoggerFactory()

get_sqlalchemy_class_logger = factory
