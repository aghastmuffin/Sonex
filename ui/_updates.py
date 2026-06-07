import os
import sys
import shutil
import tempfile
import zipfile
import requests
from packaging import version
#TODO: Ensure that script runs from root directory of the app. Enforce this.
OWNER = "aghastmuffin"
REPO = "Sonex"

LOCAL_VERSION_FILE = "version.txt"
CORE_DIR = "core"

# If you attach a zip named "app.zip" in GitHub Releases, this works automatically
ASSET_NAME_PREFERRED = None  # e.g. "app.zip" (optional)

def get_local_version():
    try:
        with open(LOCAL_VERSION_FILE, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        with open(LOCAL_VERSION_FILE, "w+") as f:
            f.write("0.0.0")
        return "0.0.0"


def get_latest_release():
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/releases/latest"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()


def pick_asset(release_json):
    assets = release_json.get("assets", [])

    if not assets:
        raise RuntimeError("No assets found in GitHub release")

    if ASSET_NAME_PREFERRED:
        for a in assets:
            if a["name"] == ASSET_NAME_PREFERRED:
                return a["browser_download_url"]

    # fallback: first asset
    return assets[0]["browser_download_url"]


def download_file(url):
    tmp = tempfile.mktemp(suffix=".zip")

    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            shutil.copyfileobj(r.raw, f)

    return tmp


def backup_core():
    if not os.path.exists(CORE_DIR):
        return None

    backup_dir = CORE_DIR + "_backup"
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)

    shutil.copytree(CORE_DIR, backup_dir)
    return backup_dir


def restore_backup(backup_dir):
    if backup_dir and os.path.exists(backup_dir):
        if os.path.exists(CORE_DIR):
            shutil.rmtree(CORE_DIR)
        shutil.copytree(backup_dir, CORE_DIR)


def extract_zip(zip_path):
    tmp_dir = tempfile.mkdtemp()

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(tmp_dir)

    # assumes GitHub release zip structure: repo-tag/
    root = os.path.join(tmp_dir, os.listdir(tmp_dir)[0])
    return root


def replace_core(new_root):
    new_core = os.path.join(new_root, "core")

    if not os.path.exists(new_core):
        raise RuntimeError("Downloaded release missing /core folder")

    if os.path.exists(CORE_DIR):
        shutil.rmtree(CORE_DIR)

    shutil.copytree(new_core, CORE_DIR)

def check_and_update():
    """
    Returns:
        True  -> updated + restart required
        False -> no update needed
    """

    local_version = get_local_version()

    release = get_latest_release()
    remote_version = release["tag_name"]

    if version.parse(remote_version) <= version.parse(local_version):
        return False

    print(f"[Updater] Updating {local_version} → {remote_version}")

    backup = None

    try:
        backup = backup_core()

        asset_url = pick_asset(release)
        zip_path = download_file(asset_url)

        extracted = extract_zip(zip_path)
        replace_core(extracted)

        # update version file
        with open(LOCAL_VERSION_FILE, "w") as f:
            f.write(remote_version)

        print("[Updater] Update successful")
        return True

    except Exception as e:
        print(f"[Updater] Update failed: {e}")

        # rollback
        restore_backup(backup)

        return False

if __name__ == "__main__":
    updated = check_and_update()

    if updated:
        print("Restart your app")
        sys.exit(0)
    else:
        print("Already up to date")