Developers
=====
Kaipy developers will need to use a different setup process then users.  This section will go over the steps needed to setup a development environment for Kaipy.  The main difference here is that you not use the `pip install` command to install Kaipy.  Instead you will need to clone the repository, install the dependencies, and setup the environment variables.

Clone repository
---------
The first step is to clone the Kaipy repository.  You can do this by running the following command in your terminal:
```bash
git clone https://wiltbemj@bitbucket.org/aplkaiju/kaipy.git
```

Install python dependencies
---------
Next you will need to install the dependencies for Kaipy.  You can do this by running the following command in your terminal after navigating to the Kaipy repository:
```bash
conda create --name kaipy-repo python=3.8
conda activate kaipy-repo
pip install -r requirements.txt
```

Setup environment variables
---------
Finally you will need to setup the environment variables for Kaipy.  You can do this by running the following command in your terminal after navigating to the Kaipy repository:

For bash shell:
```bash
cd kaipy/scripts
source setupEnvironment.sh
python checkkaipypath.py
```
For csh shell:
```csh 
cd kaipy/scripts
source setupEnvironment.csh
python checkkaipypath.py
```
The `checkkaipypath.py` script will check that the environment variables have been set correctly, by printing out the directory beginning with `kaipy` that is in your `PYTHONPATH`.
