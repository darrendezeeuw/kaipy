from setuptools import setup, find_packages

setup(
    name='kaipy',
    version='1.0.6',
    description='Python software for CGS MAGE and other Kaiju models',
    author='Kaiju team',
    author_email='wiltbemj@ucar.edu',
    url='https://bitbucket.org/aplkaiju/kaipy/src/master/',
    packages=find_packages(),
    python_requires='>=3.8,<=3.10',
    install_requires=[
        'alive_progress',
        'astropy',
        # 'cartopy',
        'cdasws',
        'configparser',
        'h5py',
        'jupyterlab',
        'matplotlib',
        'pandas',
        'progressbar',
        'pyhdf',
        'pyspedas',
        'pytest',
        'slack_sdk',
        'spacepy',
        'sphinx-rtd-theme',
        'sunpy',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD 3-Clause',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    entry_points={
        'console_scripts': [
            'cda2wind=kaipy.scripts.preproc.cda2wind:main',
            'genLFM=kaipy.scripts.preproc.genLFM:main',
            'genRCM=kaipy.scripts.preproc.genRCM:main',
            'XMLGenerator=kaipy.scripts.preproc.XMLGenerator:main'            
        ]
    }
)
