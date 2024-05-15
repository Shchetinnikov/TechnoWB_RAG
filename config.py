import os
from dotenv import load_dotenv

load_dotenv()

SERVER_HOST = os.environ.get("SERVER_HOST")
SERVER_PORT = os.environ.get("SERVER_PORT")

URL_GEN = os.environ.get("URL_GEN")
XRapidAPI_Key_GEN = os.environ.get("XRapidAPI_Key_GEN")
XRapidAPI_Host_GEN = os.environ.get("XRapidAPI_Host_GEN")

URL_TRANSL = os.environ.get("URL_TRANSL")
XRapidAPI_Key_TRANSL = os.environ.get("XRapidAPI_Key_TRANSL")
XRapidAPI_Host_TRANSL = os.environ.get("XRapidAPI_Host_TRANSL")
