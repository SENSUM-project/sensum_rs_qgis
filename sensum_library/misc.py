'''
---------------------------------------------------------------------------------
                                    misc.py
---------------------------------------------------------------------------------
Created on May 13, 2013
Last modified on Mar 09, 2014

Author(s): Mostapha Harb - Daniele De Vecchi 
           University of Pavia - Remote Sensing Laboratory / EUCENTRE Foundation

Contact: daniele.devecchi03@universitadipavia.it
         mostapha.harb@eucentre.it

Description: This module includes miscellaneous functions related to vector data
             processing, multi-processing and extraction of secondary vulnerability
             indicators (e.g., building height, alignment, regularity).

---------------------------------------------------------------------------------
Project: Framework to integrate Space-based and in-situ sENSing for dynamic 
         vUlnerability and recovery Monitoring (SENSUM)

Co-funded by the European Commission under FP7 (Seventh Framework Programme)
THEME [SPA.2012.1.1-04] Support to emergency response management
Grant agreement no: 312972

---------------------------------------------------------------------------------
License: This program is free software; you can redistribute it and/or modify
         it under the terms of the GNU General Public License as published by
         the Free Software Foundation; either version 2 of the License, or
         (at your option) any later version.
---------------------------------------------------------------------------------
'''
import config
import os
import sys
import subprocess
import osgeo.osr
import osgeo.ogr
import numpy as np
from scipy import ndimage
import shutil
import multiprocessing
from multiprocessing import Pool
import ephem
import math
from collections import Counter
from operator import itemgetter
from conversion import *
from PyQt4 import QtCore, QtGui

if os.name == 'posix':
    separator = '/'
else:
    separator = '\\'

def call_multiprocess(process,parameters_list,first_segment,last_segment):
    
    #TODO: Please add description block.
    
    processors = multiprocessing.cpu_count()
    pool = Pool(processes=processors)
    interval = (last_segment-first_segment+1)/processors
    result_list=[]
    #print parameters_list
    if processors == 2:
        parameters_first = []
        parameters_second = []
        
        #print len(parameters_list)
        #print len(parameters_list[0]),len(parameters_list[1]),len(parameters_list[2])
        for i in range(0,len(parameters_list)):
            parameters_first.append(parameters_list[i])
            parameters_second.append(parameters_list[i])
        
        start_first = int(first_segment)
        end_first = int(first_segment+interval)
        #print start_first,end_first
        start_second = int(end_first)
        end_second = int(last_segment+1)
        #print start_second,end_second
        
        parameters_first.append(start_first)
        parameters_first.append(end_first)
        parameters_second.append(start_second)
        parameters_second.append(end_second)
        result_list = pool.map(process,((parameters_first),(parameters_second)))
        
    if processors == 4:
        parameters_first = []
        parameters_second = []
        parameters_third = []
        parameters_fourth = []
        
        for i in range(0,len(parameters_list)):
            parameters_first.append(parameters_list[i])
            parameters_second.append(parameters_list[i])
            parameters_third.append(parameters_list[i])
            parameters_fourth.append(parameters_list[i])
        
        start_first = int(first_segment)
        end_first = int(first_segment+interval)
        #print start_first,end_first
        start_second = int(end_first)
        end_second = int(end_first+interval)
        
        start_third = int(end_second)
        end_third = int(end_second+interval)
        
        start_fourth = int(end_third)
        end_fourth = int(last_segment+1)
        
        parameters_first.append(start_first)
        parameters_first.append(end_first)
        parameters_second.append(start_second)
        parameters_second.append(end_second)
        parameters_third.append(start_third)
        parameters_third.append(end_third)
        parameters_fourth.append(start_fourth)
        parameters_fourth.append(end_fourth)
        result_list = pool.map(process,((parameters_first),(parameters_second),(parameters_third),(parameters_fourth)))
    
    return result_list