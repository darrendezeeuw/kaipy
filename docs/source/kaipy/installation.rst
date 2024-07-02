Installation
============

Custom
------

Conda Installation
~~~~~~~~~~~~~~~~~~~

1. Download Miniconda Installer

   - Visit the `Miniconda Downloads <https://docs.conda.io/en/latest/miniconda.html>`_ page.
   - Select the installer for Python 3.x as per your requirements.

2. Open a Terminal

   - Open a terminal window on Linux by pressing ``Ctrl + Alt + T``, or search for "Terminal" in the applications menu.

3. Navigate to Download Directory

   .. code-block:: bash

       cd ~/Downloads

   Change to the directory where the installer was downloaded, usually the ``Downloads`` directory.

4. Make the Installer Executable

   - Make the downloaded script executable. Replace ``Miniconda3-latest-Linux-x86_64.sh`` with the actual downloaded file name.

     .. code-block:: bash

         chmod +x Miniconda3-latest-Linux-x86_64.sh

5. Run the Installer

   - Execute the installer script and follow the on-screen instructions.

     .. code-block:: bash

         ./Miniconda3-latest-Linux-x86_64.sh

   - You'll need to approve the license agreement and choose the installation location.

6. Initialize Conda

   - After installation, initialize Miniconda to add Conda to your PATH.

7. Close and Reopen Your Terminal

   - To apply the changes, close and reopen your terminal window.

Creating Environment
~~~~~~~~~~~~~~~~~~~~~

Create python environment.  We support using python 3.8 through python 3.10.

.. note::

   The name of the conda environment in this example is ``kaipy``.

.. code-block:: bash

    conda create --name kaipy python=3.8

Installing kaipy
~~~~~~~~~~~~~~~~~~~

To install kaipy, run the following command:

.. code-block:: bash

    pip install kaipy