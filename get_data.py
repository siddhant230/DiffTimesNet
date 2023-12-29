import requests, zipfile, io
import os

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download&confirm=1"

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

def unzip_and_save(saved_file_path, target_path):
    z = zipfile.ZipFile(io.BytesIO(open(saved_file_path, "rb").read()))
    z.extractall(target_path)


def main(data_id, target_path):
    destination = "out.zip"
    print(f"dowload {data_id} to {destination}")
    download_file_from_google_drive(data_id, destination)
    print(f"unzipping and saving {destination}")
    unzip_and_save(destination, target_path)
    print(f"Remove {destination}")
    os.remove(destination)


if __name__ == "__main__":
    data_urls_zip_id = "1pmXvqWsfUeXWCMz5fqsP8WLKXR5jxY8z"
    target_path = "."
    main(data_id=data_urls_zip_id, target_path=target_path)