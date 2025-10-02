import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm

# Base URL
BASE_URL = "http://203.135.4.150:3333/images/"

# Only these folders/files will be downloaded (folder name can be with or without trailing slash)
TARGETS = {
    "2025-06-30",
    "2025-06-29",
    "2025-06-28",
    "2025-06-27",
    "2025-06-26",
    "2025-06-25",
    "2025-06-24",
    "2025-06-23",
    "2025-06-22",
    "2025-06-21",
    "2025-06-20",
    "2025-06-19",
    "2025-06-18",
    "2025-06-17",
    "2025-06-16",
    "2025-06-15",
    "2025-06-14",
    "2025-06-13",
    "2025-06-12",
    "2025-06-11",
    "2025-06-10",
    "2025-06-09",
    "2025-06-08",
    "2025-06-07",
    "2025-06-06",
    "2025-06-05",
    "2025-06-04",
    "2025-06-03",
    "2025-06-02",
    "2025-06-01",
}

# Create output folder
os.makedirs("downloaded_images", exist_ok=True)

def download_file(file_url, save_path):
    try:
        response = requests.get(file_url, stream=True, timeout=10)
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))
        with open(save_path, 'wb') as file, tqdm(
            desc=save_path,
            total=total,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
    except Exception as e:
        print(f"[ERROR] Could not download {file_url}: {e}")

def crawl_directory(current_url, save_dir):
    try:
        response = requests.get(current_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for link in soup.find_all("a"):
            href = link.get("href")
            if href in ["../", "/", None]:
                continue
            full_url = urljoin(current_url, href)
            local_path = os.path.join(save_dir, href.strip("/"))
            if href.endswith("/"):
                os.makedirs(local_path, exist_ok=True)
                crawl_directory(full_url, local_path)
            else:
                download_file(full_url, local_path)
    except Exception as e:
        print(f"[ERROR] Could not crawl {current_url}: {e}")

# Main logic
for item in TARGETS:
    full_url = urljoin(BASE_URL, item + "/")  # Ensure it's treated as a directory
    save_path = os.path.join("downloaded_images", item.strip("/"))
    print(f"\n[INFO] Processing folder: {item}")
    os.makedirs(save_path, exist_ok=True)
    crawl_directory(full_url, save_path)
