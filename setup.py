import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PyLMD",
    version="1.0.4",
    author="Lin, Qun-Wei",
    author_email="r07922164@csie.ntu.edu.tw",
    description="Jonathan S. Smith. The local mean decomposition and its application to EEG perception data. Journal of the Royal Society Interface, 2005, 2(5): 443-454",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shownlin/PyLMD",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
