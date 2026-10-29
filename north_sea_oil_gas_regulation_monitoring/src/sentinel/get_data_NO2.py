from datetime import datetime, timedelta
import json
from pathlib import Path

from sentinelhub import (
    CRS,
    BBox,
    MimeType,
    SentinelHubRequest,
    bbox_to_dimensions,
)

try:
    from .paths import ensure_raw_directory
    from .sentinel_config import SENTINEL5P_CDSE, load_cdse_config
except ImportError:
    from paths import ensure_raw_directory
    from sentinel_config import SENTINEL5P_CDSE, load_cdse_config


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

RAW_DIRECTORY = ensure_raw_directory("NO2")


# ---------------------------------------------------------------------
# CDSE OAuth credentials
# ---------------------------------------------------------------------

config = load_cdse_config(require_credentials=False)


# ---------------------------------------------------------------------
# Sentinel-5P collection
#
# The standard DataCollection.SENTINEL5P definition contains the retired
# CREODIAS URL. Create a copy that explicitly uses CDSE.
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Region and output dimensions
# ---------------------------------------------------------------------

bbox = BBox(
    bbox=[-2, 52, 4, 62],
    crs=CRS.WGS84,
)

image_size = bbox_to_dimensions(
    bbox,
    resolution=1000,
)


# ---------------------------------------------------------------------
# Evaluation script
#
# The first output band is NO2.
# The second output band is the valid-data mask.
# ---------------------------------------------------------------------

evalscript = """
const band = "NO2";

function setup() {
  return {
    input: [band, "dataMask"],
    output: {
      bands: 2,
      sampleType: "FLOAT32",
    },
  };
}

function evaluatePixel(samples) {
  let ret = [samples[band]];
  ret.push(samples.dataMask);
  return ret;
}
"""


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def create_request(time_interval):
    return SentinelHubRequest(
        data_folder=str(RAW_DIRECTORY),
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=SENTINEL5P_CDSE,
                time_interval=time_interval,
            )
        ],
        responses=[
            SentinelHubRequest.output_response(
                "default",
                MimeType.TIFF,
            )
        ],
        bbox=bbox,
        size=image_size,
        config=config,
    )


def is_valid_tiff(file_path):
    """
    Return True only when the file has a TIFF or BigTIFF signature.

    This prevents HTML responses saved as response.tiff from being
    treated as completed downloads.
    """

    if not file_path.is_file():
        return False

    if file_path.stat().st_size < 8:
        return False

    try:
        with file_path.open("rb") as file:
            signature = file.read(4)

        return signature in {
            b"II*\x00",  # Little-endian TIFF
            b"MM\x00*",  # Big-endian TIFF
            b"II+\x00",  # Little-endian BigTIFF
            b"MM\x00+",  # Big-endian BigTIFF
        }

    except OSError:
        return False


def extract_start_date(request_file):
    """
    Extract YYYY-MM-DD from an existing Sentinel Hub request.json.

    Both older and newer sentinelhub-py cache structures are supported.
    """

    with request_file.open("r", encoding="utf-8") as file:
        request_data = json.load(file)

    request_section = request_data.get(
        "request",
        request_data,
    )

    payload = (
        request_section.get("payload")
        or request_section.get("post_values")
    )

    if payload is None:
        raise KeyError(
            "Could not find payload or post_values in request.json"
        )

    start_time = payload[
        "input"
    ][
        "data"
    ][0][
        "dataFilter"
    ][
        "timeRange"
    ][
        "from"
    ]

    return start_time[:10]


# ---------------------------------------------------------------------
# Scan the existing cache
# ---------------------------------------------------------------------

cached_dates = set()

valid_tiff_count = 0
invalid_tiff_count = 0
incomplete_folder_count = 0
unreadable_request_count = 0

for cache_directory in RAW_DIRECTORY.iterdir():
    if not cache_directory.is_dir():
        continue

    request_file = cache_directory / "request.json"
    response_file = cache_directory / "response.tiff"

    if not request_file.is_file():
        incomplete_folder_count += 1
        continue

    if not response_file.is_file():
        incomplete_folder_count += 1
        continue

    if not is_valid_tiff(response_file):
        invalid_tiff_count += 1
        continue

    try:
        date = extract_start_date(request_file)
        cached_dates.add(date)
        valid_tiff_count += 1

    except Exception:
        unreadable_request_count += 1


# ---------------------------------------------------------------------
# Requested date range
#
# END_DATE is exclusive. The final interval begins on 2026-12-31.
# ---------------------------------------------------------------------

START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2027, 1, 1)

requested_dates = []

current_date = START_DATE

while current_date < END_DATE:
    requested_dates.append(current_date)
    current_date += timedelta(days=1)


missing_dates = [
    date
    for date in requested_dates
    if date.date().isoformat() not in cached_dates
]


# ---------------------------------------------------------------------
# Report the cache before doing any downloading
# ---------------------------------------------------------------------

print(
    f"Found {len(cached_dates)} of "
    f"{len(requested_dates)} requested dates in the cache."
)

print(f"Valid cached TIFF files: {valid_tiff_count}")
print(f"Missing dates:           {len(missing_dates)}")
print(f"Invalid TIFF/HTML files: {invalid_tiff_count}")
print(f"Incomplete folders:      {incomplete_folder_count}")
print(f"Unreadable requests:     {unreadable_request_count}")
print()


# ---------------------------------------------------------------------
# Credentials are only needed if files are missing
# ---------------------------------------------------------------------

if missing_dates:
    if not config.sh_client_id or not config.sh_client_secret:
        raise RuntimeError(
            "CDSE OAuth credentials were not found.\n"
            "Add CDSE_CLIENT_ID and CDSE_CLIENT_SECRET "
            "to the .env file."
        )


# ---------------------------------------------------------------------
# Download only missing dates
# ---------------------------------------------------------------------

downloaded_count = 0
skipped_count = 0
failed_count = 0

for current_date in missing_dates:
    following_date = current_date + timedelta(days=1)

    time_interval = (
        current_date.date().isoformat(),
        following_date.date().isoformat(),
    )

    request = create_request(time_interval)

    output_path = (
        RAW_DIRECTORY
        / request.get_filename_list()[0]
    )

    # Check the exact CDSE cache path before downloading.
    if is_valid_tiff(output_path):
        print(f"{time_interval}: already downloaded — skipped")
        skipped_count += 1
        continue

    try:
        request.save_data(
            redownload=False,
            max_threads=1,
            raise_download_errors=True,
        )

        if is_valid_tiff(output_path):
            print(f"{time_interval}: downloaded")
            downloaded_count += 1
        else:
            print(
                f"{time_interval}: response was not a valid TIFF"
            )
            failed_count += 1

    except Exception as error:
        print(
            f"{time_interval}: download failed — {error}"
        )
        failed_count += 1


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

print()
print("Finished")
print(f"Initially cached: {len(cached_dates)}")
print(f"Newly downloaded: {downloaded_count}")
print(f"Skipped:          {skipped_count}")
print(f"Failed:           {failed_count}")
print(f"Directory:        {RAW_DIRECTORY}")
