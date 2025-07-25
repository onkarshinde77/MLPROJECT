from setuptools import setup ,find_packages
from typing import List

hypen_e = '-e .'

def get_dependency(file:str)->List[str]:
    lib=[]
    with open(file,'r') as f:
        for line in f:
            line = line.strip() # ingnoring \n space
            if line and not line.startswith('#'):
                lib.append(str(line))
        if hypen_e in lib:
            lib.remove(hypen_e)
    return lib

# sort version
# def get_dependency(file: str) -> List[str]:
#     with open(file, 'r') as f:
#         return [line.strip() for line in f if line.strip() and not line.startswith('#')]


setup(
    name='mlproject',
    version='0.0.1',
    author='onkar shinde',
    author_email='shindeonkar704@gmail.com',
    packages = find_packages(),
    install_requires = get_dependency('requirements.txt'),
    
)