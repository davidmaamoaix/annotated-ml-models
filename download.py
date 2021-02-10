import os
import wget
import shutil
import zipfile


def coco128():
    url = 'https://github.com/ultralytics/'\
        'yolov5/releases/download/v1.0/coco128.zip'
    base_dir = 'dataset'

    if not os.path.exists('dataset'):
        os.mkdir('dataset')

    if os.path.exists(os.path.join(base_dir, 'coco128')):
        print('Coco128 already exists, skipping.')
        return

    print('Downloading coco128.')
    filename = wget.download(url, out=base_dir)

    print('\nDownload finished. Unzipping.')

    with zipfile.ZipFile(filename, 'r') as f:
        f.extractall(base_dir)

    os.remove(filename)

    shutil.move('dataset/coco128/images/train2017', 'dataset/coco128')
    os.rename('dataset/coco128/train2017', 'dataset/coco128/images')

    shutil.move('dataset/coco128/labels/train2017', 'dataset/coco128')
    os.rename('dataset/coco128/train2017', 'dataset/coco128/labels')

    os.rmdir('dataset/coco128/images/')
    os.rmdir('dataset/coco128/labels/')