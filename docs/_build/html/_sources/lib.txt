The Developer's Guide
======================

-------------
Introduction
-------------

**General Knowledge**
    Sensum Earth Observation Tools are written in Python. They make use of Gdal library to read and write rasters files as arrays, handled by Scipy, Numpy, Python math and OpenCV libraries. 
    OGR is used to read, write and handle vector files. OTB and skimage are used for classification and segmentation algorithms. The Plugin UI is written in PyQT using the qgis.core library to interact with QGIS interface.

**Structure**
    Sensum Library is composed by modules divided according to their scope. To import a Sensum module you have to import the config.py script first in order to automatically set all the necessary variables and configuration settings:
    ::
        import config
    You can import a sensum library module with the following line:
    ::
        from sensum_library.MODULE_NAME import *

----------------------
Conversion, Features
----------------------

**Conventions**
    This is the main module to read/write and handle rasters. It is composed by a set of functions with a set of convenction:
        * 0 value is used to set defaults value in a lot of functions parameters.
        * *band_list*: is a N-D list of matrices containing pixel values read with gdal ReadAsArray method, where N = number of bands
        * *input/output_raster*: path of raster file
        * *input/output_shape*: path of vector file

**Main Functions**
    Main functions are `read_image <modules.html#sensum_library.conversion.read_image>`_ , `write_image <modules.html#sensum_library.conversion.write_image>`_ used to read and write rasters files.

------------------------------
Classification, Segmentation
------------------------------

Classification and segmentation modules include functions to define hubs between python and OTB and skimage in order to take advantage of classification and segmentation algorithms. In some scripts or plugin parts a system call for external compiled OTB executable is included in order to get the process progress bar. This is achieved using the executeOtb function:

.. literalinclude:: ../sensum.py
    :linenos:
    :language: python
    :lines: 114-135

----------------------------------------
Preprocess, Secondary Indicators
----------------------------------------

Contains a series of scripts and tools functions.

------------
Multi
------------

**Intro**
Multiprocess module is dedicated to multiprocess application. Python provides a powerful built-in multiprocess library; the Sensum multiprocess module uses this library to implement multiprocess applications through a container Class. This solution was specifically designed to provide clean and easy implementation also for non-expert developers.

**Description**

    Multi() class defines N concurrent processes with N = number of CPU core * 2.

**Example**
    We want to implement with multiprocess a simple function for get the sum between a number and his double within a range from 0 to 100:
    ::
        def task(a):
            return a*2+a

        for i in range(101):
            print task(i)

    To implement task() as multiprocess we need to reclass it as a callable Class:
    ::
        class Task(object):
            def __init__(self, a):
                self.index = a
                self.a = a
            def __call__(self):
                return self.index,self.a*2+self.a

    Unlike function declaration, we need to return also the index since results will not be sorted because multiprocess execution is asynchronous. Now we can implement Task() into the for statement:
    ::
        from sensum_library.multiprocess import *
        MyMulti = Multi()
        for i in xrange(101):
            MyMulti.put(Task(i))
        MyMulti.kill()

    Use the following lines to get the results, sort and print them:
    ::
        results = [ MyMulti.result() for i in xrange(101) ]
        results.sort()
        results = [ data for i,data in results ]

        for data in results:
            print data


Following the complete code example:

.. literalinclude:: ../sensum_library/multi.py
    :linenos:
    :language: python
    :lines: 108-128

--------
Scripts
--------

For more info about the script system `click here <user.html#introduction>`_

**Progress Bar**

This is a class used to draw a textual process bar compatible with `plugin parser <lib.html#plugin>`_:

.. literalinclude:: ../scripts/utils.py
    :linenos:
    :language: python
    :lines: 25-45

Use is simple:
::
    n_list = range(100,200)
    status = Bar(len(n_list),"Power N")
    for i,value in n_list
        status(i)
        print value*value

-------
Plugin
-------

**PyQGIS Developer Cookbook**
    For more details go to `http://docs.qgis.org/2.0/en/docs/pyqgis_developer_cookbook/ <http://docs.qgis.org/2.0/en/docs/pyqgis_developer_cookbook/>`_ 

**Call Scripts**
To call scripts, the Sensum Plugin uses a function also able to parse the textual progress bar generated by the script:

.. literalinclude:: ../sensum.py
    :linenos:
    :language: python
    :lines: 77-112
