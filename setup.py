from setuptools import setup, find_packages

setup(
    name='py2dataset',
    version='0.1',
    packages=find_packages(),
    url='https://github.com/jeffmeloy/py2dataset',
    license='MIT',
    author='Jeff Meloy',
    author_email='jeffmeloy@gmail.com',
    description='A tool to convert Python code into structured datasets',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=[
        'ast', 
        'networkx', 
        'ctransformers', 
        'yaml',
        # add more dependencies here
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
    python_requires='>=3.8',
)