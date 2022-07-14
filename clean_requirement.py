"""
The existing requirements.txt is generated from conda.
This script fixes it for pip.

Local package is not installed.
Run:
$ pip install .

Author: Luke Chang (xcha011@aucklanduni.ac.nz)
Date:   14/07/2022
"""

BLACKLIST = [
    'blas',
    'ca-certificates',
    'cuda',
    'freetype',
    'icu',
    'jpeg',
    'libpng',
    'libtiff',
    'libuv',
    'libwebp',
    'lz4-c',
    'matplotlib-base',
    'numpy-base',
    'numpy',
    'nvita',
    'openssl',
    'pip',
    'pyqt',
    'python',
    'pytorch',
    'pywin32',
    'pywinpty',
    'qt',
    'sip',
    'sqlite',
    'tk',
    'torch',
    'tzdata',
    'vc',
    'vs2015_runtime',
    'win',
    'xz',
    'zlib',
]

def is_in_blacklist(line):
    for item in BLACKLIST:
        if line.startswith(item):
            return True
    return False

def main():
    newfile = ''
    with open('requirements.txt') as file:
        lines = file.readlines()
        for line in lines:
            # Remove last part after "=" sign
            newline = '=='.join(line.split('=')[:-1]) + '\n'
            if len(newline) > 1 and not is_in_blacklist(newline):
                newfile += newline

    with open('requirements_recovered.txt', 'w') as file:
        file.write(newfile)

if __name__ == '__main__':
    main()