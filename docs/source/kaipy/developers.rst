Developers
================================================
Kaipy developers will need to use a different setup process then users.  This section will go over the steps needed to setup a development environment for Kaipy.  The main difference is that developers will need to clone the Kaipy repository and install using the `pip install -e .` command.  

Clone repository
------------------------------------------------

The first step is to clone the Kaipy repository.  You can do this by running the following command in your terminal:

.. code-block:: bash

    git clone https://github.com/JHUAPL/kaipy.git

Install Editable Kaipy
------------------------------------------------

Next you will need to install the dependencies for Kaipy.  You can do this by running the following commands in your terminal after navigating to the Kaipy repository:

.. code-block:: bash

    conda create --name kaipy-repo-3.12 python=3.12
    conda activate kaipy-repo-3.12
    pip install -e .

This will install Kaipy in editable mode, allowing you to make changes to the code and have them reflected immediately without needing to reinstall.  

.. warning::
    
 If any changes are made to either of the `setup.py` or `pyproject.toml` files, or new dependencies are added to the `requirements.txt` file, the kaipy package must be reinstalled with the `pip install -e .` command.
