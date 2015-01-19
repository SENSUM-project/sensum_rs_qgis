'''
/***************************************************************************
 Sensum
                                 A QGIS plugin
 Sensum QGIS Plugin
                              -------------------
        begin                : 2014-05-27
        copyright            : (C) 2014 by Eucentre
        email                : dgaleazzo@gmail.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************
'''
import sys
import os
import subprocess
import shutil

if os.name == "posix":
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    lines = str(subprocess.check_output("whereis otb", shell=True)).split("\n")
    otb_path = (line[5:]+"/applications" for line in lines if "lib" in line).next()
    sys.path.append(otb_path)
    logos_path = "{}/.sensum/".format(os.path.expanduser("~"))
else:
    bit = ("64" if os.path.isdir("C:/OSGeo4W64") else "")
    osgeopath = "C:/OSGeo4W{}/".format(bit)
    os.environ["ITK_AUTOLOAD_PATH"] = "C:/OSGeo4W{}/apps/orfeotoolbox/applications".format(bit)
    logos_path = osgeopath+"bin/.sensum/"
if not os.path.exists(logos_path):
    os.mkdir(logos_path)
os.chdir(osgeopath+'bin/')
if (os.path.isfile("{}sensum.png".format(logos_path)) and os.path.isfile("{}unipv.png".format(logos_path)) and os.path.isfile("{}eucentre.png".format(logos_path))) == 0:
    current_dir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    paths = (current_dir+"/icons/unipv.png",current_dir+"/icons/sensum.png",current_dir+"/icons/eucentre.png")
    for path in paths: shutil.copy(path, logos_path)
