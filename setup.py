from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = "-e ."

def get_requirements(file_path:str) -> List[str]:
    """
    Args:
        file_path (str): path to requirements.txt file

    Returns:
        List[str]: return list of requirements
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements


setup(
    name="end-to-end-ml-project",
    version="0.0.1",
    author="fethi",
    author_email="azibi.fethi99@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)