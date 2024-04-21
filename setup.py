from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="py2dataset",
    version="0.5",
    author="Jeff Meloy",
    author_email="jeffmeloy@gmail.com",
    description="A tool to generate structured datasets from Python source code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jeffmeloy/py2dataset",
    py_modules=["get_code_graph", "get_params", "get_python_datasets", "get_python_file_details", "py2dataset", "save_output"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=["matplotlib", "networkx", "ctransformers", "PyYaml", "GitPython", ],
    entry_points={"console_scripts": ["py2dataset = py2dataset:main"]},
    packages=["py2dataset"],
    package_dir={"py2dataset": ".\\"},
)
