import os
from pathlib import Path

from nvita.utils import open_json

PATH_ROOT = Path(os.getcwd()).absolute()
print(PATH_ROOT)


if __name__ == '__main__':
    my_metadata = open_json(os.path.join(
        PATH_ROOT, 'experiment', 'metadata.json'))
    print(my_metadata)

    att = 'something'
    print('something:', att in my_metadata['attacks'])

    att = 'fgsm'
    print('fgsm:', att in my_metadata['attacks'])
