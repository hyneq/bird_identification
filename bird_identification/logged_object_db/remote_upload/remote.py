import requests
import requests_toolbelt.sessions

from ..model import ClassLoggedObjectModel

class RemoteAPI:

    session: requests.Session

    def __init__(self, session: requests.Session):
        self.session = session


    def upload(self, objects: list[ClassLoggedObjectModel]) -> list[int]:
        data = []

        for o in objects:
            data.append({
                "class_name": o.class_name,
                "start_time": str(o.start_time),
                "end_time": str(o.end_time)
            })

        response = self.session.post("upload", json=data)
        response.raise_for_status()

        return response.json()


def get_remote_api(base_url: str, username: str, password: str):

    session = requests_toolbelt.sessions.BaseUrlSession(base_url)

    session.auth = (username, password)

    return RemoteAPI(session)
