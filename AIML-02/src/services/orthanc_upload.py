import io
import zipfile
import requests

ZIP_PATH = r"C:\Users\Admin\Documents\Ai_Medical_Imaging\data\PCIR_98890234_20010101_7.zip"
ORTHANC_URL = "http://localhost:8042"

uploaded = 0
failed = 0

with zipfile.ZipFile(ZIP_PATH, "r") as archive:

    files = [
        item for item in archive.infolist()
        if not item.is_dir()
        and not item.filename.startswith("__MACOSX/")
    ]

    print("Files found in ZIP:", len(files))

    for item in files:
        try:
            dicom_data = archive.read(item)

            response = requests.post(
                f"{ORTHANC_URL}/instances",
                data=dicom_data,
                headers={"Content-Type": "application/dicom"},
                timeout=60
            )

            if response.status_code in (200, 202):
                uploaded += 1
            else:
                failed += 1

        except Exception:
            failed += 1

        if (uploaded + failed) % 10 == 0:
            print(f"Processed: {uploaded + failed}/{len(files)}")

print("\nUpload completed.")
print("Successfully uploaded:", uploaded)
print("Failed:", failed)