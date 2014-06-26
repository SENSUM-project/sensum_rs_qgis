import sys
import os
import subprocess
import shutil

if os.name == "posix":
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    lines = str(subprocess.check_output("whereis otb", shell=True)).split("\n")
    otb_path = (line[5:]+"/applications" for line in lines if "lib" in line).next()
    sys.path.append(otb_path)
    install_path = "{}/.sensum/".format(os.path.expanduser("~"))
else:
    bit = ("64" if os.path.isdir("C:/OSGeo4W64") else "")
    sys.path.append("{}apps/Python27/Lib/site-packages".format(install_path))
    sys.path.append("{}apps/orfeotoolbox/python".format(install_path))
    os.environ["PATH"] = os.environ["PATH"] + ";C:/OSGeo4W{}/bin;C:/Python27".format(bit)
    install_path = "C:/OSGeo4W{}/bin/.qgis/".format(bit)
if not os.path.exists(install_path):
    os.mkdir(install_path)
if (os.path.isfile("{}sensum.png".format(install_path)) and os.path.isfile("{}unipv.png".format(install_path)) and os.path.isfile("{}eucentre.png".format(install_path))) == 0:
    current_dir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    paths = (current_dir+"/uis/unipv.png",current_dir+"/uis/sensum.png",current_dir+"/uis/eucentre.png")
    for path in paths: shutil.copy(path, install_path)
