import os
import requests
import tarfile
from tqdm import tqdm
from pathlib import Path

MPIMNIST_ZENODO_ID = "12799417"

def input_yes_no(default='y'):
    """
    Prompt the user for a yes (y) or no (n) input.

    The user is repeatedly asked for input until a valid response is given.

    Args:
        default (str, optional): The default value to return if the user provides empty input. 
                                 Must be either 'y' or 'n'. Defaults to 'y'.

    Returns:
        bool: True if the user input is 'y[es]', False if the user input is 'n[o]'.
    """
    def _input():
        inp = input()
        inp = inp.lower()
        if inp in ['y', 'yes']:
            inp = 'y'
        elif inp in ['n', 'no']:
            inp = 'n'
        elif inp == '':
            inp = default
        else:
            print('please input y[es] or n[o]')
            return None
        return inp

    inp = _input()
    while inp not in ['y', 'n']:
        inp = _input()

    return inp == 'y'

def download_file(url, dest_folder):
    """
    Download a file from a specified URL to a destination folder.

    This function downloads a file from the given URL and saves it to the specified 
    destination folder. If the destination folder does not exist, it is created.

    Args:
        url (str): The URL of the file to download.
        dest_folder (str): The folder where the downloaded file will be saved.

    Returns:
        str: The path to the downloaded file.
    """
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
        
    local_filename = os.path.join(dest_folder, url.split('/')[-1])
    response = requests.get(url, stream=True)
    
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kilobyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    
    with open(local_filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    
    progress_bar.close()
    
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR: Download failed!")
    
    return local_filename

def extract_tar_gz(file_path, extract_folder):
    """
    Extract a .tar.gz file to a specified folder.

    This function extracts the contents of a .tar.gz file to the given extraction folder.
    If the extraction folder does not exist, it is created.

    Args:
        file_path (str): The path to the .tar.gz file to be extracted.
        extract_folder (str): The folder where the contents will be extracted.
    """
    if not os.path.exists(extract_folder):
        os.makedirs(extract_folder)
        
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path=extract_folder)

def extract_without_top_directory(tar_ref, target_folder):
    """
    Extract files from a `.tar.gz` archive while removing the top-level directory.

    Args:
        tar_ref (tarfile.TarFile): The opened tar file object.
        target_folder (str): The folder where files should be extracted.
    """
    for member in tar_ref.getmembers():
        member_path = Path(member.name)

        stripped_path = Path(*member_path.parts[1:])

        if member.isreg(): 
            target_path = os.path.join(target_folder, stripped_path)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with tar_ref.extractfile(member) as source, open(target_path, 'wb') as target:
                target.write(source.read())

def download_and_unpack_zenodo_files(zenodo_id, extract_folder):
    """
    Download and extract zipped files from a Zenodo record.

    This function downloads all files associated with a given Zenodo record ID and 
    extracts their contents into the specified extraction folder. Each extracted file
    is placed in a subfolder named after the file (without extensions).

    Args:
        extract_folder (str): The folder where the downloaded files will be extracted.
    """
    zenodo_url = f"https://zenodo.org/record/{zenodo_id}/files/"
    response = requests.get(f"https://zenodo.org/api/records/{zenodo_id}")
    if response.status_code != 200:
        print(f"Error fetching Zenodo record: {response.status_code}")
        return
    
    data = response.json()
    files = data['files']
    
    for file_info in files:
        file_url = file_info['links']['self']
        file_name = file_info['key']

        if not file_name.endswith('.tar.gz'):
            print(f"Skipping non-compressed file: {file_name}")
            continue

        folder_name = os.path.splitext(os.path.splitext(file_name)[0])[0]
        target_folder = os.path.join(extract_folder, folder_name)
        print(f"Downloading {file_url} ...")
        downloaded_file = download_file(file_url, extract_folder)
        print(f"Extracting {downloaded_file} into {target_folder} ...")
        with tarfile.open(downloaded_file, 'r:gz') as tar_ref:
            extract_without_top_directory(tar_ref, target_folder)

        os.remove(downloaded_file)
        print(f"Done with {downloaded_file}")

def download_and_unpack_mpimnist(extract_folder):
    """
    Download and extract all files from MPI-MNIST Zenodo record.

    Args:
        extract_folder (str): The folder where the downloaded files will be extracted.
    """
    extract_folder_mpimnist = os.path.join(extract_folder,'MPI-MNIST')
    download_and_unpack_zenodo_files(MPIMNIST_ZENODO_ID, extract_folder_mpimnist)
    return