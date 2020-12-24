import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="papersec", # Replace with your own username
    version="0.0.1",
    author="Kim Minseong",
    author_email="devget43@gmail.com",
    description="Recurrent State Navigator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/papersec/rsn",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Ubuntu 20.04",
    ],
    python_requires='>=3.8',
)