from pathlib import Path

from src.services.dicom_service import convert_dicom_to_png

BASE_DIR = Path(__file__).resolve().parents[1]

dicom_path = BASE_DIR / "samples" / "sample.dcm"
output_path = BASE_DIR / "samples" / "dicom_converted.png"

result = convert_dicom_to_png(dicom_path, output_path)

print("DICOM conversion successful!")
print("Output:", result)