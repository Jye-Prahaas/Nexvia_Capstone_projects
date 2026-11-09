import numpy as np
import pydicom
import nibabel as nib

from pathlib import Path
from PIL import Image
from pydicom.pixel_data_handlers.util import (
    apply_modality_lut,
    apply_voi_lut
)




def read_dicom(dicom_path):

    dicom_path = Path(dicom_path)

    if not dicom_path.exists():
        raise FileNotFoundError(
            f"DICOM file not found: {dicom_path}"
        )

    ds = pydicom.dcmread(str(dicom_path))

    if "PixelData" not in ds:
        raise ValueError("DICOM file does not contain pixel data.")

    if int(getattr(ds, "NumberOfFrames", 1)) != 1:
        raise ValueError("Multi-frame DICOM is not supported.")

    image = ds.pixel_array

    image = apply_modality_lut(image, ds)

    try:
        image = apply_voi_lut(image, ds)
    except Exception:
        pass

    image = image.astype(np.float32)

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        image = image.max() - image

    low = np.percentile(image, 1)
    high = np.percentile(image, 99)

    if high <= low:
        raise ValueError("Invalid DICOM pixel intensity range.")

    image = np.clip(image, low, high)
    image = (image - low) / (high - low)

    return (image * 255).astype(np.uint8)



def convert_dicom_to_png(dicom_path, output_path):

    image = read_dicom(dicom_path)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(image).save(output_path)

    return str(output_path)


def read_dicom_series(series_directory):

    series_directory = Path(series_directory)

    if not series_directory.is_dir():
        raise NotADirectoryError(
            f"DICOM series directory not found: {series_directory}"
        )

    slices = []

    for file_path in series_directory.iterdir():

        if not file_path.is_file():
            continue

        try:
            ds = pydicom.dcmread(str(file_path))

            if "PixelData" not in ds:
                continue

            if int(getattr(ds, "NumberOfFrames", 1)) != 1:
                continue

            if getattr(ds, "Modality", "") != "CT":
                continue

            slices.append((ds, file_path))

        except Exception:
            continue

    if len(slices) < 2:
        raise ValueError(
            "At least two single-frame CT DICOM slices are required."
        )

    
    series_uids = {
        str(ds.SeriesInstanceUID)
        for ds, _ in slices
        if hasattr(ds, "SeriesInstanceUID")
    }

    if len(series_uids) != 1:
        raise ValueError(
            "Directory contains multiple CT series. "
            "Provide one series at a time."
        )

    
    first_ds = slices[0][0]

    orientation = np.asarray(
        first_ds.ImageOrientationPatient,
        dtype=np.float64
    )

    row_direction = orientation[:3]
    column_direction = orientation[3:]

    normal = np.cross(row_direction, column_direction)

    slices.sort(
        key=lambda item: np.dot(
            np.asarray(item[0].ImagePositionPatient, dtype=np.float64),
            normal
        )
    )

    reference = slices[0][0]

    rows = int(reference.Rows)
    columns = int(reference.Columns)

    pixel_spacing = np.asarray(
        reference.PixelSpacing,
        dtype=np.float64
    )

    frame_uid = getattr(reference, "FrameOfReferenceUID", None)

    expected_orientation = orientation

    positions = []
    pixel_arrays = []

    for ds, _ in slices:

        if int(ds.Rows) != rows or int(ds.Columns) != columns:
            raise ValueError("Inconsistent CT image dimensions.")

        if not np.allclose(
            np.asarray(ds.PixelSpacing, dtype=np.float64),
            pixel_spacing,
            atol=1e-4
        ):
            raise ValueError("Inconsistent CT pixel spacing.")

        current_orientation = np.asarray(
            ds.ImageOrientationPatient,
            dtype=np.float64
        )

        if not np.allclose(
            current_orientation,
            expected_orientation,
            atol=1e-4
        ):
            raise ValueError("Inconsistent CT slice orientations.")

        if frame_uid is not None and getattr(
            ds, "FrameOfReferenceUID", None
        ) != frame_uid:
            raise ValueError("Inconsistent CT frame of reference.")

        position = np.asarray(
            ds.ImagePositionPatient,
            dtype=np.float64
        )

        positions.append(position)

        pixels = ds.pixel_array

       
        pixels = apply_modality_lut(pixels, ds)

        pixel_arrays.append(
            pixels.astype(np.float32)
        )

    positions = np.asarray(positions)

    projected_positions = positions @ normal

    differences = np.diff(projected_positions)

    if np.any(differences <= 0):
        raise ValueError("Invalid CT slice ordering.")

    slice_spacing = float(np.median(differences))

    if not np.allclose(
        differences,
        slice_spacing,
        rtol=0.02,
        atol=0.05
    ):
        raise ValueError(
            "Irregular CT slice spacing detected. "
            "Resampling is required before segmentation."
        )

    volume = np.stack(pixel_arrays, axis=-1)

   
    affine = np.eye(4)

    affine[:3, 0] = column_direction * pixel_spacing[0]
    affine[:3, 1] = row_direction * pixel_spacing[1]
    affine[:3, 2] = normal * slice_spacing
    affine[:3, 3] = positions[0]

    return {
        "volume": volume,
        "affine": affine,
        "spacing": (
            float(pixel_spacing[0]),
            float(pixel_spacing[1]),
            slice_spacing
        ),
        "slices": len(pixel_arrays),
        "rows": rows,
        "columns": columns
    }



def convert_dicom_series_to_nifti(
    series_directory,
    output_path
):

    result = read_dicom_series(series_directory)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nifti_image = nib.Nifti1Image(
        result["volume"],
        result["affine"]
    )

    nib.save(nifti_image, str(output_path))

    return {
        "output_path": str(output_path),
        "shape": result["volume"].shape,
        "spacing": result["spacing"],
        "slices": result["slices"]
    }