from sqlalchemy.engine import create_engine, Engine
from sqlalchemy.orm import sessionmaker, scoped_session

from ..model import ClassLoggedObjectModel

class LocalDB:
    engine: Engine

    def __init__(self, engine: Engine):
        super().__init__()

        self.engine = engine
        self.Session = scoped_session(sessionmaker(engine))


    def get_session(self):
        return self.Session.begin()


    def get_not_synced(self):
        session = self.Session()
        return list(session.query(ClassLoggedObjectModel).filter_by(remote_id=None).all())


    def update_synced(self, objects: list[ClassLoggedObjectModel], remote_ids: list[int]):
        if len(objects) != len(remote_ids):
            raise ValueError("Objects and remote_ids lengths differ")
        
        for i, o in enumerate(objects):
            o.remote_id = remote_ids[i]


def get_local_db(db_url: str):
    engine = create_engine(db_url)

    return LocalDB(engine)
