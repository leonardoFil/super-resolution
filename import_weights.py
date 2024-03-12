import requests
import tarfile
import os

def import_tar_gz_from_url(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)

def unzip_tar_gz(file_path):
    with tarfile.open(file_path, 'r:gz') as file:
        file.extractall('\\'.join(file_path.split('\\')[:-1]))
        
if __name__ == '__main__':
    url = 'https://martin-krasser.de/sisr/weights-edsr-16-x4.tar.gz'
    save_path = 'submodules\sr\weights-edsr-16-x4.tar.gz'
    import_tar_gz_from_url(url, save_path)
    if os.path.exists(save_path):
        os.remove(save_path)