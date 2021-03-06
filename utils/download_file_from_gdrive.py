import requests


def download_file_from_gdrive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = _get_confirm_token(response)

    print('Donwloading from Google Drive ...')
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
        _save_response_content(response, destination)
        print('Sucess downloading.')
    else:
        print('Failed to download.')
        exit(-1)


def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def _save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
