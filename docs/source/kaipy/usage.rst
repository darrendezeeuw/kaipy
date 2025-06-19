Usage
================================================

The Kaipy package comes with a variety of tools to help you load and analyze data from the MAGE model. This page will walk you through the basic usage of the package.  This includes using the API to load datasets and create plots as well as using the command line interface to run scripts.

Kaipy API Package
------------------------------------------------

We will go over the basics of loading datasets and creating plots using the Kaipy API package.  You can try this out for yourself using our `Google Colab notebook <https://colab.research.google.com/drive/1Y559nAryHyX5R9wgqSLvZ-87QRmazGLD#scrollTo=0bOWk7gnNzd2>`_.

Loading datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to load data for the GAMERA and MIX components of the MAGE model you will need to import the following modules:

.. code-block:: python

    import kaipy.gamera.magsphere as msph 
    import kaipy.remix.remix as remix
    import os

Start by importing the magnetosphere data. You will need to set the `fdir` variable to the directory containing the output from a MAGE magnetosphere run. The `ftag` variable will need to be set to the name that identifies the MAGE ouput.  For example

.. code-block:: python

    basedir = '/content/gdrive/MyDrive/MAGEColab'
    fdir = os.path.join(basedir,'GrossREU')
    ftag = 'GrossREUSlim'

Now we use the msph portion of the kaipy package to setup a pipeline to the magnetosphere data. For this example we will set the step we want to display to the final step of the run.

.. code-block:: python
    
    gsph = msph.GamsphPipe(fdir,ftag,doFast=False)
    nstep = gsph.sFin

Importing the ionospheric data from REMIX follow the same format as the import of the magnetospheric data with the added requirement of specifying which hemisphere, e.g. NORTH or SOUTH, that you want to plot.

.. code-block:: python
    
    mixFiles = os.path.join(fdir,"%s.mix.h5"%(ftag))
    ion = remix.remix(mixFiles,nstep)
    ion.init_vars('NORTH')

Plots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The msphViz portion of the kaipy library has numerous routines for visualizing the magnetosphere. Here will demostrate usage of plotXY and plotXZ which make plots of the equatorial and meridional cuts of the magnetosphere. For both of these routines and others in system you need to provide the instance of the magnetosphere object, the step number you want to display, the extent of the domain, and two sets of axes. The first axes object is for the plot and the second is for the colorbar. For plotXZ we will use the minimal set arguments which results in a plot density. For the plotXY example we will specify options so that we end up with Vx on Red/Blue color palette with a mid point normalization.

First the plotXZ example

.. code-block:: python
    
    xyBds = [-100,20,-60,60]
    figSz = (8,8)
    fig = plt.figure(figsize=figSz)
    gs = fig.add_gridspec(2,1,height_ratios=[20,1],hspace=0.2)
    Ax1 = fig.add_subplot(gs[0,0])
    AxC1 = fig.add_subplot(gs[1,0])
    data = mviz.plotXZ(gsph,nstep,xyBds,Ax1,AxC1,vMin=0,vMax=50)

.. image:: /_static/plotXZExample.png
    :alt: Example XZ plot of Density
    :width: 400px
    :align: center
    
And for the plotXY example

.. code-block:: python
    
    figSz = (8,8)
    fig = plt.figure(figsize=figSz)
    gs = fig.add_gridspec(2,1,height_ratios=[20,1],hspace=0.2)
    Ax1 = fig.add_subplot(gs[0,0])
    AxC1 = fig.add_subplot(gs[1,0])
    data = mviz.plotXY(gsph,nstep,xyBds,Ax1,AxC1,var='Vx',midp=True,cmap='RdBu_r')

.. image:: /_static/plotXYExample.png
    :alt: Example XY plot of Vx
    :width: 400px
    :align: center

The mix object includes an extensive plotting routine that has the capability for numerous variables with excellent choices for the color tables. It also takes advantage of the mix object's ability to calculate derived quanties, such as magnetic perturbations and electric fields. Unlike the magnetosphere plotting routines it has the option to take a gridspec object instead of an axes object. It also has the option be made an inset plot so that it can be easily combined with a magnetosphere plot.

.. code-block:: python
    
    ion.plot('current')

.. image:: /_static/ionExample.png
    :alt: Example plot of ionospheric current and potential
    :width: 400px
    :align: center

Kaipy Command Line Interface
------------------------------------------------

The Kaipy package also comes with a command line interface that allows you to run scripts to analyze MAGE model data.  The CLI is a great way to automate the analysis of large datasets.  The CLI is run from the terminal and has a variety of options to customize the analysis. 

A complete list of the available scripts can be found at the `Scripts documentation <https://kaipy-docs.readthedocs.io/en/latest/scripts.html>`_.

The quicklook directory has numerous scripts that can be used to generate plots and movies of the MAGE model output.  For example the `msphpic.py` command makes a summary movie of the magnetosphere while the `mixpic.py` command makes a summary movie of the ionosphere.

.. autoprogram:: msphpic:create_command_line_parser()
    :prog: msphpic.py

.. autoprogram:: mixpic:create_command_line_parser()
    :prog: mixpic.py