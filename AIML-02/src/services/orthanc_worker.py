import asyncio
import json
import os
import time
from pathlib import Path

import requests
from requests.auth import HTTPBasicAuth


ORTHANC_URL = os.getenv(
    "ORTHANC_URL",
    "http://localhost:8042"
).rstrip("/")

FASTAPI_URL = os.getenv(
    "FASTAPI_URL",
    "http://127.0.0.1:8000"
).rstrip("/")

ORTHANC_USERNAME = os.getenv("ORTHANC_USERNAME")
ORTHANC_PASSWORD = os.getenv("ORTHANC_PASSWORD")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_FILE = PROJECT_ROOT / "data" / "orthanc_results.json"

POLL_INTERVAL = 10
STABLE_SECONDS = 30
PAGE_SIZE = 100

pending_studies = {}


def get_auth():
    if ORTHANC_USERNAME and ORTHANC_PASSWORD:
        return HTTPBasicAuth(
            ORTHANC_USERNAME,
            ORTHANC_PASSWORD
        )
    return None


def get_study_id(instance_id):
    """Resolve an Orthanc instance to its parent study."""

    instance = requests.get(
        f"{ORTHANC_URL}/instances/{instance_id}",
        auth=get_auth(),
        timeout=30
    )
    instance.raise_for_status()

    series_id = instance.json()["ParentSeries"]

    series = requests.get(
        f"{ORTHANC_URL}/series/{series_id}",
        auth=get_auth(),
        timeout=30
    )
    series.raise_for_status()

    return series.json()["ParentStudy"]


def save_result(study_id, result):
    """Persist prediction metadata without patient demographics."""

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    if RESULTS_FILE.exists():
        try:
            records = json.loads(
                RESULTS_FILE.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError):
            records = {}
    else:
        records = {}

    records[study_id] = {
        "scan_id": result.get("scan_id"),
        "filename": result.get("filename"),
        "tumour_detected": result.get("tumour_detected"),
        "tumour_voxels": result.get("tumour_voxels"),
        "tumour_volume_ml": result.get("tumour_volume_ml"),
        "confidence": result.get("confidence"),
        "device": result.get("device"),
        "processed_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    RESULTS_FILE.write_text(
        json.dumps(records, indent=4),
        encoding="utf-8"
    )


def process_study(study_id):
    """Download a complete study and submit it to FastAPI."""

    print("Processing an Orthanc study...")

    archive = requests.get(
        f"{ORTHANC_URL}/studies/{study_id}/archive",
        auth=get_auth(),
        timeout=180
    )
    archive.raise_for_status()

    response = requests.post(
        f"{FASTAPI_URL}/predict/dicom-series",
        files={
            "file": (
                "orthanc_study.zip",
                archive.content,
                "application/zip"
            )
        },
        timeout=900
    )

    response.raise_for_status()

    result = response.json()
    save_result(study_id, result)

    print("Orthanc inference completed.")
    print("Prediction saved successfully.")


async def orthanc_worker():
    """Continuously monitor Orthanc for new studies."""

    cursor = 0

    print("Orthanc background worker started.")

    while True:
        try:
            response = await asyncio.to_thread(
                requests.get,
                f"{ORTHANC_URL}/changes",
                params={
                    "since": cursor,
                    "limit": PAGE_SIZE
                },
                auth=get_auth(),
                timeout=30
            )

            response.raise_for_status()
            data = response.json()

            for change in data.get("Changes", []):

                if (
                    change.get("ChangeType") == "NewInstance"
                    and change.get("ResourceType") == "Instance"
                ):
                    instance_id = change.get("ID")

                    study_id = await asyncio.to_thread(
                        get_study_id,
                        instance_id
                    )

                    pending_studies[study_id] = time.monotonic()

            cursor = max(cursor, data.get("Last", cursor))

            now = time.monotonic()

            ready = [
                study_id
                for study_id, last_seen in pending_studies.items()
                if now - last_seen >= STABLE_SECONDS
            ]

            for study_id in ready:
                try:
                    await asyncio.to_thread(
                        process_study,
                        study_id
                    )

                except Exception as error:
                    print(
                        "Orthanc study processing failed:",
                        type(error).__name__
                    )

                finally:
                    pending_studies.pop(study_id, None)

        except Exception as error:
            print(
                "Orthanc monitoring error:",
                type(error).__name__
            )

        await asyncio.sleep(POLL_INTERVAL)