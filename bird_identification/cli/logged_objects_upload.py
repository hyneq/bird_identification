import requests

from ..logged_object_db.remote_upload import get_remote_uploader

from . import CLI

class LoggedObjectsUpload(CLI):

    def init_parser(self):
        super().init_parser()

        self.parser.add_argument("local_db_url", help="The URL to local database")
        self.parser.add_argument("remote_api_url", help="Base URL of the remote API")
        self.parser.add_argument("-u", "--username", default=None, help="The username for remote API")
        self.parser.add_argument("-p", "--password", default=None, help="The password for remote API")


    def run(self):
        super().run()

        if self.args.username and not self.args.password:
            raise ValueError("Username and password must be supplied both or none")
        
        uploader = get_remote_uploader(
            local_db_url=self.args.local_db_url,
            remote_api_url=self.args.remote_api_url,
            remote_api_username=self.args.username,
            remote_api_password=self.args.password
        )

        uploaded_n = uploader.upload()
        (f"Uploaded {uploaded_n} logged objects to {self.args.remote_api_url}")


def cli_main():
    LoggedObjectsUpload().run()