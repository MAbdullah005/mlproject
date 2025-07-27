# building my application into package

from setuptools import setup,find_packages
import os

HYEPN_DOT='-e .'
def get_requirement(file_path)->list[str]:

    requirements=[]
    with open(file=file_path) as f:
        requirements=f.readlines()
        requirements=[req.replace('\n',' ') for req in requirements]
        if HYEPN_DOT in requirements:
            requirements.remove(HYEPN_DOT)
        return requirements

setup(
    name="mlproject",
    version='1.0',
    author="Abdullah Ali",
    author_email="abdullahaliofc@gmail.com",
    packages=find_packages(),
    install_requires=get_requirement('requirements.txt'),
)

if __name__=='__main__':
    print(os.getcwd)