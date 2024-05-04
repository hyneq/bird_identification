from .local import LocalDB, get_local_db
from .remote import RemoteAPI, get_remote_api

class RemoteUploader:

    local_db: LocalDB
    remote_api: RemoteAPI

    def __init__(self, local_db: LocalDB, remote_api: RemoteAPI):
        self.local_db = local_db
        self.remote_api = remote_api


    def upload(self):
        with self.local_db.get_session():
            objects = self.local_db.get_not_synced()

            if len(objects) == 0:
                return

            ids = self.remote_api.upload(objects)

            self.local_db.update_synced(objects, ids)

            return len(objects)


def get_remote_uploader(
        local_db_url: str,
        remote_api_url: str,
        remote_api_username: str, remote_api_password: str
    ):

    local_db = get_local_db(local_db_url)
    remote_api = get_remote_api(remote_api_url, remote_api_username, remote_api_password)

    return RemoteUploader(local_db, remote_api)
