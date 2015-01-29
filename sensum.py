# -*- coding: utf-8 -*-
"""
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
 ***************************************************************************/
"""
# Import the PyQt and QGIS libraries
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from qgis.core import *
# Initialize Qt resources from file resources.py
import resources_rc
# Import the code for the dialog
from sensumdialog import *
import os,sys
import os.path
import shutil
import subprocess
from scripts.utils import *


try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

def string_qmark(string):
    try: return "\""+string+"\""
    except: return None

def executeScript(command, progress=None,noerror=True):
    if os.name != "posix" and noerror:
        # Found at http://stackoverflow.com/questions/5069224/handling-subprocess-crash-in-windows
        # Don't display the Windows GPF dialog if the invoked program dies.
        # See comp.os.ms-windows.programmer.win32
        # How to suppress crash notification dialog?, Jan 14,2004 -
        # Raymond Chen's response [1]

        import ctypes
        SEM_NOGPFAULTERRORBOX = 0x0002 # From MSDN
        ctypes.windll.kernel32.SetErrorMode(SEM_NOGPFAULTERRORBOX);
        subprocess_flags = 0x8000000 #win32con.CREATE_NO_WINDOW?
    else:
        subprocess_flags = 0

    command = (os.path.dirname(os.path.abspath(__file__))+command if os.name == "posix" else 'C:/OSGeo4W64/bin/python.exe "'+os.path.dirname(os.path.abspath(__file__))+command)
    #QMessageBox.information(None, "Info", command) #DEBUG
    proc = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=subprocess_flags,
        universal_newlines=True,
        ).stdout
    for line in iter(proc.readline, ''):
        if '[*' in line:
            idx = line.find('[*')
            perc = line[idx:(idx+102)].count("*")
            status = line[line.find('STATUS: ')+8:idx]
            if perc != 0 and progress:
                progress.progressBar.setValue(perc)
                progress.label_title.setText(status)
        QtGui.qApp.processEvents()

def executeOtb(command, progress=None,label = "OTB library recalled"):
    if os.name != "posix":
        bit = ("64" if os.path.isdir("C:/OSGeo4W64") else "")
        osgeopath = "C:/OSGeo4W{}/bin/".format(bit)
        command = osgeopath + command
    #QMessageBox.information(None, "Info", command) #DEBUG
    proc = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        ).stdout
    for line in iter(proc.readline, ''):
        if '[*' in line:
            idx = line.find('[*')
            perc = int(line[idx - 4:idx - 2].strip(' '))
            if perc != 0 and progress:
                progress.progressBar.setValue(perc)
                progress.label_title.setText(label)
        QtGui.qApp.processEvents()

def segmentation_progress(input_raster,output_file,filter,paramaters=None,progress=None):
    paramaters_string = ""
    if paramaters:
        for paramater in paramaters.keys():
            value = paramaters.get(paramater)
            if value:
                paramaters_string = paramaters_string + "-filter.{}.{} {} ".format(filter,paramater,value)
    command = "otbcli_Segmentation -progress 1 {} -in {} -filter {} -mode.vector.out {}".format(paramaters_string,input_raster,filter,output_file)
    executeOtb(command, progress,filter)

def parse_input(string):
    return string.replace("[Choose from a file..] ","")

class Sensum:

    def __init__(self, iface):
        # Save reference to the QGIS interface
        self.iface = iface
        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)
        # initialize locale
        locale = QSettings().value("locale/userLocale")[0:2]
        localePath = os.path.join(self.plugin_dir, 'i18n', 'sensum_{}.qm'.format(locale))
        if os.path.exists(localePath):
            self.translator = QTranslator()
            self.translator.load(localePath)

            if qVersion() > '4.3.3':
                QCoreApplication.installTranslator(self.translator)

    def initGui(self):

        self.toolBar = self.iface.addToolBar("SENSUM")

        ######################
        ## FEATURES
        ######################
        # Create action that will start plugin configuration
        self.action_features = QAction(
            QIcon(":/plugins/sensum_plugin/icons/features.png"),
            u"Features", self.iface.mainWindow())
        # connect the action to the run method
        self.action_features.triggered.connect(self.features)
        # Add toolbar button and menu item
        self.toolBar.addAction(self.action_features)
        self.iface.addPluginToMenu(u"&SENSUM", self.action_features)

        ######################
        ## BUILDING HEIGHT
        ######################
        # Create action that will start plugin configuration
        self.action_building_height = QAction(
            QIcon(":/plugins/sensum_plugin/icons/buildingsheight.png"),
            u"Building Height", self.iface.mainWindow())
        # connect the action to the run method
        self.action_building_height.triggered.connect(self.build_height)
        # Add toolbar button and menu item
        self.toolBar.addAction(self.action_building_height)
        self.iface.addPluginToMenu(u"&SENSUM", self.action_building_height)

        ######################
        ## CO-REGISTRATION
        ######################
        # Create action that will start plugin configuration
        self.action_coregistration = QAction(
            QIcon(":/plugins/sensum_plugin/icons/coregistration.png"),
            u"Coregistration", self.iface.mainWindow())
        # connect the action to the run method
        self.action_coregistration.triggered.connect(self.coregistration)
        # Add toolbar button and menu item
        self.toolBar.addAction(self.action_coregistration)
        self.iface.addPluginToMenu(u"&SENSUM", self.action_coregistration)

        ######################
        ## FOOTPRINTS
        ######################
        # Create action that will start plugin configuration
        self.action_footprints = QAction(
            QIcon(":/plugins/sensum_plugin/icons/footprints.png"),
            u"Footprints", self.iface.mainWindow())
        # connect the action to the run method
        self.action_footprints.triggered.connect(self.footprints)
        # Add toolbar button and menu item
        self.toolBar.addAction(self.action_footprints)
        self.iface.addPluginToMenu(u"&SENSUM", self.action_footprints)

        ######################
        ## STACK SATELLITE
        ######################
        # Create action that will start plugin configuration
        self.action_stacksatellite = QAction(
            QIcon(":/plugins/sensum_plugin/icons/stacksatellite.png"),
            u"Stack Satellite", self.iface.mainWindow())
        # connect the action to the run method
        self.action_stacksatellite.triggered.connect(self.stacksatellite)
        # Add toolbar button and menu item
        self.toolBar.addAction(self.action_stacksatellite)
        self.iface.addPluginToMenu(u"&SENSUM", self.action_stacksatellite)

        ######################
        ## DENSITY
        ######################
        # Create action that will start plugin configuration
        self.action_density = QAction(
            QIcon(":/plugins/sensum_plugin/icons/density.png"),
            u"Density", self.iface.mainWindow())
        # connect the action to the run method
        self.action_density.triggered.connect(self.density)
        # Add toolbar button and menu item
        self.toolBar.addAction(self.action_density)
        self.iface.addPluginToMenu(u"&SENSUM", self.action_density)

        ######################
        ## REGULARITY
        ######################
        # Create action that will start plugin configuration
        self.action_regularity = QAction(
            QIcon(":/plugins/sensum_plugin/icons/regularity.png"),
            u"regularity", self.iface.mainWindow())
        # connect the action to the run method
        self.action_regularity.triggered.connect(self.regularity)
        # Add toolbar button and menu item
        self.toolBar.addAction(self.action_regularity)
        self.iface.addPluginToMenu(u"&SENSUM", self.action_regularity)

        ######################
        ## CHANGE DETECTION
        ######################
        # Create action that will start plugin configuration
        self.action_change = QAction(
            QIcon(":/plugins/sensum_plugin/icons/change.png"),
            u"Change", self.iface.mainWindow())
        # connect the action to the run method
        self.action_change.triggered.connect(self.change)
        # Add toolbar button and menu item
        self.toolBar.addAction(self.action_change)
        self.iface.addPluginToMenu(u"&SENSUM", self.action_change)

    def unload(self):
        self.iface.removePluginMenu(u"&SENSUM", self.action_features)
        self.iface.removeToolBarIcon(self.action_features)
        self.iface.removePluginMenu(u"&SENSUM", self.action_building_height)
        self.iface.removeToolBarIcon(self.action_building_height)
        self.iface.removePluginMenu(u"&SENSUM", self.action_coregistration)
        self.iface.removeToolBarIcon(self.action_coregistration)
        self.iface.removePluginMenu(u"&SENSUM", self.action_footprints)
        self.iface.removeToolBarIcon(self.action_footprints)
        self.iface.removePluginMenu(u"&SENSUM", self.action_stacksatellite)
        self.iface.removeToolBarIcon(self.action_stacksatellite)
        self.iface.removePluginMenu(u"&SENSUM", self.action_density)
        self.iface.removeToolBarIcon(self.action_density)
        self.iface.removePluginMenu(u"&SENSUM", self.action_regularity)
        self.iface.removeToolBarIcon(self.action_density)
        self.iface.removePluginMenu(u"&SENSUM", self.action_change)

    def changeActive(self,comboBox):
        comboBox.clear()
        comboBox.addItem(_fromUtf8(""))
        comboBox.setItemText(0, _translate("Pansharp", "[Choose from a file..]", None))
        current_layer = self.iface.mapCanvas().currentLayer()
        layers = self.iface.mapCanvas().layers()
        for i,layer in enumerate(layers):
            path = str(layer.dataProvider().dataSourceUri()).replace("|layerid=0","")
            comboBox.addItem(_fromUtf8(""))
            comboBox.setItemText(i+1, _translate("Pansharp", path, None))

    def features(self):
        # Create the dialog (after translation) and keep reference
        self.dlg_features = FeaturesDialog()
        self.changeActive(self.dlg_features.ui.comboBox_input)
        QObject.connect(self.iface.mapCanvas(), SIGNAL( "layersChanged()" ), lambda: self.changeActive(self.dlg_features.ui.comboBox_input))
        self.changeActive(self.dlg_features.ui.comboBox_training)
        QObject.connect(self.iface.mapCanvas(), SIGNAL( "layersChanged()" ), lambda: self.changeActive(self.dlg_features.ui.comboBox_training))
        dlgProgress = ProgressDialog()
        # show the dialog
        self.dlg_features.show()
        # Run the dialog event loop
        result = self.dlg_features.exec_()
        # See if OK was pressed
        if result == 1:
            ui = self.dlg_features.ui
            dlgProgress.show()
            input_image = parse_input(str(ui.comboBox_input.currentText()))
            input_shape = parse_input(str(ui.comboBox_training.currentText()))
            output_shape = str(ui.lineEdit_output.text())
            field = str(ui.lineEdit_field.text())
            spectrals = [(ui.checkBox_multiprocess,"--multi"),(ui.checkBox_maxbr,"--max_br"),(ui.checkBox_mean,"--mean"),(ui.checkBox_minbr,"--min_br"),(ui.checkBox_mode,"--mode"),(ui.checkBox_ndivistd,"--ndvi_std"),(ui.checkBox_ndvimean,"--ndvi_mean"),(ui.checkBox_std,"--std"),(ui.checkBox_weighbr,"--weigh_br")]
            textures = [(ui.checkBox_asm,"--ASM"),(ui.checkBox_contrast,"--contrast"),(ui.checkBox_correlation,"--correlation"),(ui.checkBox_dissimilarity,"--dissimilarity"),(ui.checkBox_energy,"--energy"),(ui.checkBox_homogeneity,"--homogeneity")]
            indexes_list_spectral = " ".join([index for pushButton,index in spectrals if pushButton.isChecked()])
            indexes_list_texture = " ".join([index for pushButton,index in textures if pushButton.isChecked()])
            executeScript('/scripts/test_features.py" \"{}\" \"{}\" \"{}\" \"{}\" {} {}'.format(input_image,input_shape,output_shape,field,indexes_list_spectral,indexes_list_texture),dlgProgress.ui)
            QgsMapLayerRegistry.instance().addMapLayer(QgsVectorLayer(output_shape,os.path.splitext(os.path.basename(output_shape))[0], "ogr"))
            QMessageBox.information(None, "Info", 'Done!')

    def build_height(self):
        # Create the dialog (after translation) and keep reference
        self.dlg_height = BuildHeightDialog()
        self.changeActive(self.dlg_height.ui.comboBox_input_buildings)
        QObject.connect(self.iface.mapCanvas(), SIGNAL( "layersChanged()" ), lambda: self.changeActive(self.dlg_height.ui.comboBox_input_buildings))
        self.changeActive(self.dlg_height.ui.comboBox_input_shadows)
        QObject.connect(self.iface.mapCanvas(), SIGNAL( "layersChanged()" ), lambda: self.changeActive(self.dlg_height.ui.comboBox_input_shadows))
        dlgProgress = ProgressDialog()
        # show the dialog
        self.dlg_height.show()
        # Run the dialog event loop
        result = self.dlg_height.exec_()
        # See if OK was pressed
        if result == 1:
            ui = self.dlg_height.ui
            dlgProgress.show()
            input_buildings = parse_input(str(ui.comboBox_input_buildings.currentText()))
            input_shadow = parse_input(str(ui.comboBox_input_shadows.currentText()))
            output_shape = str(ui.lineEdit_output.text())
            idfield = str(ui.lineEdit_shadow_field.text())
            window_resize = str(ui.doubleSpinBox_window_paramater.text()).replace(",",".")
            window_resize = float(window_resize)
            date = str(ui.dateTimeEdit.text())
            executeScript('/scripts/height.py" \"{}\" \"{}\" \"{}\" \"{}\" \"{}\" \"{}\"'.format(input_shadow,input_buildings,date,output_shape,idfield,window_resize),dlgProgress.ui)
            QgsMapLayerRegistry.instance().addMapLayer(QgsVectorLayer(output_shape,os.path.splitext(os.path.basename(output_shape))[0], "ogr"))
            QMessageBox.information(None, "Info", 'Done!')

    def coregistration(self):
        # Create the dialog (after translation) and keep reference
        self.dlg_coregistration = CoregistrationDialog()
        dlgProgress = ProgressDialog()
        # show the dialog
        self.dlg_coregistration.show()
        # Run the dialog event loop
        result = self.dlg_coregistration.exec_()
        # See if OK was pressed
        if result == 1:
            ui = self.dlg_coregistration.ui
            dlgProgress.show()
            target_folder = str(ui.lineEdit_tobechange.text())
            reference_path = str(ui.lineEdit_reference.text())
            options = [(ui.checkBox_surf, "--enable_SURF" ),(ui.checkBox_fft, "--enable_FFT")]
            options = " ".join([index for pushButton,index in options if pushButton.isChecked()])
            select_crop = ""
            if str(ui.comboBox_select_crop.currentText()) == "Clip":
                select_crop = "--enable_clip {} ".format(string_qmark(parse_input(ui.comboBox_input_shape.currentText())))
            elif str(ui.comboBox_select_crop.currentText()) == "Grid":
                select_crop = "--enable_grid {} {} ".format(ui.spinBox_rows.text(), ui.spinBox_cols.text())
            elif str(ui.comboBox_select_crop.currentText()) == "Unsupervised Classification":
                select_crop = "--enable_unsupervised {} ".format(ui.spinBox_nclasses.text())
            executeScript('/scripts/coregistration.py" \"{}\" \"{}\" {} {}'.format(reference_path,target_folder,select_crop,options),dlgProgress.ui)
            QMessageBox.information(None, "Info", 'Done!')

    def footprints(self):
        # Create the dialog (after translation) and keep reference
        self.dlg_footprints = FootprintsDialog()
        self.changeActive(self.dlg_footprints.ui.comboBox_pansharp)
        QObject.connect(self.iface.mapCanvas(), SIGNAL( "layersChanged()" ), lambda: self.changeActive(self.dlg_footprints.ui.comboBox_pansharp))
        self.changeActive(self.dlg_footprints.ui.comboBox_training)
        QObject.connect(self.iface.mapCanvas(), SIGNAL( "layersChanged()" ), lambda: self.changeActive(self.dlg_footprints.ui.comboBox_training))
        dlgProgress = ProgressDialog()
        # show the dialog
        self.dlg_footprints.show()
        # Run the dialog event loop
        result = self.dlg_footprints.exec_()
        # See if OK was pressed
        if result == 1:
            ui = self.dlg_footprints.ui
            dlgProgress.show()
            pansharp_file = parse_input(str(ui.comboBox_pansharp.currentText()))
            training_set = parse_input(str(ui.comboBox_training.currentText()))
            training_attribute = str(ui.lineEdit_training_field.text())
            output_shape = str(ui.lineEdit_output.text())
            checked_optimizer = ("--optimizer" if bool(ui.checkBox_filter.isChecked()) else "")
            if checked_optimizer == "--optimizer":
                executeOtb("otbcli_MeanShiftSmoothing -progress 1 -in {} -spatialr {} -ranger {} -thres {} -maxiter {} -fout {} -foutpos {}".format(pansharp_file,30,30,0.1,100,pansharp_file[:-4]+'_smooth.tif',pansharp_file[:-4]+'_sp.tif'),progress=dlgProgress.ui,label="Smooth Filter")
                pansharp_file = pansharp_file[:-4]+'_smooth.tif'
                checked_optimizer = ""
            building_classes = [str(ui.listWidget.item(index).text()) for index in xrange(ui.listWidget.count())]
            building_classes = ("-c \""+"\" \"".join(building_classes)+"\"" if len(building_classes) else "")
            executeScript('/scripts/footprints.py" \"{}\" \"{}\" \"{}\" \"{}\" {} {}'.format(pansharp_file,training_set,training_attribute,output_shape,checked_optimizer,building_classes),dlgProgress.ui)
            QgsMapLayerRegistry.instance().addMapLayer(QgsVectorLayer(output_shape,os.path.splitext(os.path.basename(output_shape))[0], "ogr"))
            QMessageBox.information(None, "Info", 'Done!')

    def stacksatellite(self):
        # Create the dialog (after translation) and keep reference
        self.dlg_stacksatellite = StackSatelliteDialog()
        dlgProgress = ProgressDialog()
        # show the dialog
        self.dlg_stacksatellite.show()
        # Run the dialog event loop
        result = self.dlg_stacksatellite.exec_()
        # See if OK was pressed
        if result == 1:
            ui = self.dlg_stacksatellite.ui
            dlgProgress.show()
            sat_folder = str(ui.lineEdit_satellite_folder.text())
            segmentation_name = str(ui.comboBox_segmentation.currentText())
            n_classes = str(ui.spinBox_nclasses.text())
            ref_dir = ("--ref_dir "+string_qmark(str(ui.lineEdit_reference_directory.text())) if bool(ui.checkBox_reference_diretory.isChecked()) else "")
            input_shapefile = ("--restrict_to_city "+string_qmark(parse_input(str(ui.comboBox_input_shapefile.currentText()))) if bool(ui.checkBox_restrict_to_city.isChecked()) else "")
            options = [(ui.checkBox_coregistration , "--coregistration"),(ui.checkBox_builtup_index, "--builtup_index_method" ),(ui.checkBox_pca_index, "--pca_index_method"),(ui.checkBox_pca_classification , "--pca_classification_method"),(ui.checkBox_dissimilarity, "--dissimilarity_method" ),(ui.checkBox_pca_ob, "--pca_ob_method" )]
            options = " ".join([index for pushButton,index in options if pushButton.isChecked()])
            segmentation_paramaters = list()
            if segmentation_name == "Edison":
                segmentation_paramaters.append(int(ui.lineEdit_edison_radius.text()))
                segmentation_paramaters.append(int(ui.lineEdit_edison_range.text()))
                segmentation_paramaters.append(int(ui.lineEdit_edison_size.text()))
                segmentation_paramaters.append(int(ui.lineEdit_edison_scale.text()))
            elif segmentation_name == "Meanshift":
                segmentation_paramaters.append(int(ui.lineEdit_meanshift_radius.text()))
                segmentation_paramaters.append(float(ui.lineEdit_meanshift_range.text()))
                segmentation_paramaters.append(float(ui.lineEdit_meanshift_threshold.text()))
                segmentation_paramaters.append(int(ui.lineEdit_meanshift_iterations.text()))
                segmentation_paramaters.append(int(ui.lineEdit_meanshift_minsize.text()))
            string_segmentation_paramaters = " ".join([str(paramater) for paramater in segmentation_paramaters])
            executeScript('/scripts/stacksatellite.py" \"{}\" \"{}\" \"{}\" {} {} {} --segmentation_paramaters {}'.format(
                sat_folder,
                segmentation_name,
                n_classes,
                ref_dir,
                input_shapefile,
                options,
                string_segmentation_paramaters),dlgProgress.ui)
            QMessageBox.information(None, "Info", 'Done!')

    def density(self):
        # Create the dialog (after translation) and keep reference
        self.dlg_density = DensityDialog()
        self.changeActive(self.dlg_density.ui.comboBox_building_shape)
        QObject.connect(self.iface.mapCanvas(), SIGNAL( "layersChanged()" ), lambda: self.changeActive(self.dlg_density.ui.comboBox_building_shape))
        dlgProgress = ProgressDialog()
        # show the dialog
        self.dlg_density.show()
        # Run the dialog event loop
        result = self.dlg_density.exec_()
        # See if OK was pressed
        if result == 1:
            ui = self.dlg_density.ui
            dlgProgress.show()
            buildingShape = parse_input(str(ui.comboBox_building_shape.currentText()))
            radius = str(ui.doubleSpinBox_radius.text()).replace(",",".")
            radius = float(radius)
            outputShape = str(ui.lineEdit_output_shapefile.text())
            if os.path.isfile(outputShape): os.remove(outputShape)
            executeScript('/scripts/density.py" \"{}\" \"{}\" \"{}\"'.format(buildingShape,radius,outputShape),dlgProgress.ui)
            QgsMapLayerRegistry.instance().addMapLayer(QgsVectorLayer(outputShape,os.path.splitext(os.path.basename(outputShape))[0], "ogr"))
            QMessageBox.information(None, "Info", 'Done!')
    
    def change(self):
        # Create the dialog (after translation) and keep reference
        self.dlg_change = ChangeDialog()
        dlgProgress = ProgressDialog()
        # show the dialog
        self.dlg_change.show()
        # Run the dialog event loop
        result = self.dlg_change.exec_()
        # See if OK was pressed
        if result == 1:
            ui = self.dlg_change.ui
            dlgProgress.show()
            sat_folder = str(ui.lineEdit_tobechange.text())
            extraction = ( "Dissimilarity" if str(ui.comboBox_extraction.currentText()) == "Dissimilarity-Based" else "PCA")
            field = str(ui.lineEdit_field.text())
            spatial_filter = (" --spatial_filter " if bool(ui.checkBox_spatial_filter.isChecked()) else "")
            executeScript('/scripts/change_detection.py" \"{}\" \"{}\" \"{}\" \"{}\"'.format(sat_folder,extraction,field,spatial_filter),dlgProgress.ui)
            output_shape = sat_folder+"/change_detection.shp"
            QgsMapLayerRegistry.instance().addMapLayer(QgsVectorLayer(output_shape,os.path.splitext(os.path.basename(output_shape))[0], "ogr"))
            QMessageBox.information(None, "Info", 'Done!')

    def regularity(self):
        # Create the dialog (after translation) and keep reference
        self.dlg_regularity = RegularityDialog()
        self.changeActive(self.dlg_regularity.ui.comboBox_building_shape)
        QObject.connect(self.iface.mapCanvas(), SIGNAL( "layersChanged()" ), lambda: self.changeActive(self.dlg_regularity.ui.comboBox_building_shape))
        dlgProgress = ProgressDialog()
        # show the dialog
        self.dlg_regularity.show()
        # Run the dialog event loop
        result = self.dlg_regularity.exec_()
        # See if OK was pressed
        if result == 1:
            ui = self.dlg_regularity.ui
            dlgProgress.show()
            buildingShape = parse_input(str(ui.comboBox_building_shape.currentText()))
            outputShape = str(ui.lineEdit_output_shapefile.text())
            if os.path.isfile(outputShape): os.remove(outputShape)
            executeScript('/scripts/regularity.py" \"{}\" \"{}\"'.format(buildingShape,outputShape),dlgProgress.ui)
            QgsMapLayerRegistry.instance().addMapLayer(QgsVectorLayer(outputShape,os.path.splitext(os.path.basename(outputShape))[0], "ogr"))
            QMessageBox.information(None, "Info", 'Done!')
