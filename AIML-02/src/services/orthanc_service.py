import os
import requests
from requests.auth import HTTPBasicAuth

ORTHANC_URL = os.getenv(
    "ORTHANC_URL",
    "http://localhost:8042"
).rstrip("/")


ORTHANC_USERNAME = os.getenv("ORTHANC_USERNAME")
ORTHANC_PASSWORD = os.getenv("ORTHANC_PASSWORD")


def get_auth():
    if ORTHANC_USERNAME and ORTHANC_PASSWORD:
        return HTTPBasicAuth(
            ORTHANC_USERNAME,
            ORTHANC_PASSWORD
        )
    return None


def get_changes(since=0):
    response = requests.get(
        f"{ORTHANC_URL}/changes",
        params={
            "since": since,
            "limit": 100
        },
        auth=get_auth(),
        timeout=30
    )

    response.raise_for_status()
    return response.json()


def get_study(study_id):
    response = requests.get(
        f"{ORTHANC_URL}/studies/{study_id}",
        auth=get_auth(),
        timeout=30
    )

    response.raise_for_status()
    return response.json()


def download_study(study_id, destination):
    response = requests.get(
        f"{ORTHANC_URL}/studies/{study_id}/archive",
        auth=get_auth(),
        timeout=120
    )

    response.raise_for_status()

    with open(destination, "wb") as file:
        file.write(response.content)

    return destination


def get_system_info():
    response = requests.get(
        f"{ORTHANC_URL}/system",
        auth=get_auth(),
        timeout=30
    )

    response.raise_for_status()
    return response.json()