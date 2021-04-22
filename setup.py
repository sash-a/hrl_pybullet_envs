from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hrl_pybullet_envs",
    version="0.1.2",
    author="Sasha Abramowitz",
    author_email="reallysasha@gmail.com",
    description="Locomotion HRL envs in pybullet",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sash-a/hrl_pybullet_envs",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires='>=3.6',
    include_package_data=True
)
