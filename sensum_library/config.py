import sys
import os
import subprocess
import shutil

if sys.platform == "linux2":
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    lines = str(subprocess.check_output("whereis otb", shell=True)).split("\n")
    otb_path = (line[5:]+"/applications" for line in lines if "lib" in line).next()
    sys.path.append(otb_path)
    logos_path = "{}/.sensum/".format(os.path.expanduser("~"))
elif sys.platform == 'darwin':
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    logos_path = "{}/.sensum/".format(os.path.expanduser("~"))
else:
    bit = ("64" if os.path.isdir("C:/OSGeo4W64") else "")
    osgeopath = "C:/OSGeo4W{}/".format(bit)
    sys.path.append("{}apps/Python27/Lib/site-packages".format(osgeopath))
    sys.path.append("{}apps/orfeotoolbox/python".format(osgeopath))
    #os.environ["PATH"] = os.environ["PATH"] + ";C:/OSGeo4W{}/bin;C:/Python27".format(bit)
    os.environ["ITK_AUTOLOAD_PATH"] = "C:/OSGeo4W{}/apps/orfeotoolbox/applications".format(bit)
    logos_path = osgeopath+"bin/.sensum/"
if not os.path.exists(logos_path):
    os.mkdir(logos_path)
os.chdir(osgeopath+'bin/')
if (os.path.isfile("{}sensum.png".format(logos_path)) and os.path.isfile("{}unipv.png".format(logos_path)) and os.path.isfile("{}eucentre.png".format(logos_path))) == 0:
    current_dir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    paths = (current_dir+"/icons/unipv.png",current_dir+"/icons/sensum.png",current_dir+"/icons/eucentre.png")
    for path in paths: shutil.copy(path, logos_path)
