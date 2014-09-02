# -*- coding: utf-8 -*-
"""
/***************************************************************************
 SensumDialog
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

from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import *
from PyQt4.QtGui import *

from uis.ui_progress import Ui_Progress
from uis.ui_pansharp import Ui_Pansharp
from uis.ui_classification import Ui_Classification
from uis.ui_segmentation import Ui_Segmentation
from uis.ui_features import Ui_Features
from uis.ui_build_height import Ui_BuildHeight
from uis.ui_coregistration import Ui_Coregistration
from uis.ui_footprints import Ui_Footprints
from uis.ui_stacksatellite import Ui_StackSatellite
from uis.ui_density import Ui_Density
from uis.ui_temporal import Ui_Temporal

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

class ProgressDialog(QtGui.QDialog, Ui_Progress):
    def __init__(self):
        QtGui.QDialog.__init__(self)
        # Set up the user interface from Designer.
        # After setupUI you can access any designer object by doing
        # self.<objectname>, and you can use autoconnect slots - see
        # http://qt-project.org/doc/qt-4.8/designer-using-a-ui-file.html
        # #widgets-and-dialogs-with-auto-connect
        self.ui = Ui_Progress()
        self.ui.setupUi(self)

        self.ui.progressBar.setValue( 0 )

class PansharpDialog(QtGui.QDialog, Ui_Pansharp):
    def __init__(self):
        QtGui.QDialog.__init__(self)
        self.ui = Ui_Pansharp()
        self.ui.setupUi(self)
        QObject.connect(self.ui.comboBox_multiband, SIGNAL("activated(const QString&)"), self.setPath_multiband)
        QObject.connect(self.ui.comboBox_panchromatic, SIGNAL("activated(const QString&)"), self.setPath_panchromatic)
        QObject.connect(self.ui.pushButton_output, SIGNAL("clicked()"), self.setPath_output)

    def setPath_multiband(self):
        if self.ui.comboBox_multiband.currentIndex() == 0:
            fileName = QFileDialog.getOpenFileName(self,"Open Image", "~/","Image Files (*.tiff *.tif)");
            if fileName !="":
                self.ui.comboBox_multiband.setItemText(0, _translate("Pansharp", "[Choose from a file..] "+fileName, None))
            else:
                self.ui.comboBox_multiband.setItemText(0, _translate("Pansharp", "[Choose from a file..]", None))
    def setPath_panchromatic(self):
        if self.ui.comboBox_multiband.currentIndex() == 0:
            fileName = QFileDialog.getOpenFileName(self,"Open Image", "~/","Image Files (*.tif)");
            if fileName !="":
                self.ui.comboBox_panchromatic.setItemText(0, _translate("Pansharp", "[Choose from a file..] "+fileName, None))
            else:
                self.ui.comboBox_panchromatic.setItemText(0, _translate("Pansharp", "[Choose from a file..]", None))
    def setPath_output(self):
        fileName = QFileDialog.getSaveFileName(self,"Output Image", "~/","Image Files (*.tiff *.tif)");
        if fileName !="":
            self.ui.lineEdit_output.setText(fileName)

class ClassificationDialog(QtGui.QDialog, Ui_Classification):
    def __init__(self):
        QtGui.QDialog.__init__(self)
        self.ui = Ui_Classification()
        self.ui.setupUi(self)
        self.ui.frame_supervised.show()
        self.ui.frame_unsupervised.hide()
        #QObject.connect(self.ui.pushButton_input, SIGNAL("clicked()"), self.setPath_input)
        QObject.connect(self.ui.comboBox_input, SIGNAL("activated(const QString&)"), self.setPath_input)
        QObject.connect(self.ui.pushButton_output, SIGNAL("clicked()"), self.setPath_output)
        QObject.connect(self.ui.pushButton_training, SIGNAL("clicked()"), self.setPath_training)
        QObject.connect(self.ui.comboBox_supervised, SIGNAL("currentIndexChanged(const QString&)"), self.select_options_frame)
    def setPath_input(self):
        if self.ui.comboBox_input.currentIndex() == 0:
            fileName = QFileDialog.getOpenFileName(self,"Open Image", "~/","Image Files (*.tiff *.tif)");
            if fileName !="":
                self.ui.comboBox_input.setItemText(0, _translate("Pansharp", "[Choose from a file..] "+fileName, None))
            else:
                self.ui.comboBox_input.setItemText(0, _translate("Pansharp", "[Choose from a file..]", None))
    def setPath_output(self):
        fileName = QFileDialog.getSaveFileName(self,"Output Image", "~/","Image Files (*.tiff *.tif)");
        if fileName !="":
            self.ui.lineEdit_output.setText(fileName)
    def setPath_training(self):
        fileName = QFileDialog.getOpenFileName(self,"Input training file", "~/","ESRI Shapefile Files (*.shp)");
        if fileName !="":
            self.ui.lineEdit_training.setText(fileName)
    def select_options_frame(self):
        option = str(self.ui.comboBox_supervised.currentText())
        if option == "Supervised":
            self.ui.frame_supervised.show()
            self.ui.frame_unsupervised.hide()
        else:
            self.ui.frame_supervised.hide()
            self.ui.frame_unsupervised.show()

class SegmentationDialog(QtGui.QDialog, Ui_Segmentation):
    def __init__(self):
        QtGui.QDialog.__init__(self)
        self.ui = Ui_Segmentation()
        self.ui.setupUi(self)
        self.ui.groupBox_optimizer.hide()
        self.ui.groupBox_options.show()
        self.options = [("Baatz",self.ui.frame_baatz),("Edison",self.ui.frame_edison),("Meanshift",self.ui.frame_meanshift),("Morphological Profiles",self.ui.frame_morphological),("Region Growing",self.ui.frame_region),("Watershed",self.ui.frame_watershed),("Felzenszwalb",self.ui.frame_felzenszwalb)]
        self.show_option()
        #QObject.connect(self.ui.pushButton_input, SIGNAL("clicked()"), self.setPath_input)
        QObject.connect(self.ui.comboBox_input, SIGNAL("activated(const QString&)"), self.setPath_input)
        QObject.connect(self.ui.pushButton_output, SIGNAL("clicked()"), self.setPath_output)
        QObject.connect(self.ui.pushButton_optimizer_input, SIGNAL("clicked()"), self.setPath_optimizer_input)
        QObject.connect(self.ui.checkBox_optimizer, SIGNAL("stateChanged(int)"), self.select_optimizer_frame)
        QObject.connect(self.ui.comboBox_method, SIGNAL("currentIndexChanged(const QString&)"), self.show_option)
    def hide_options(self):
        for method,frame in self.options:
            frame.hide()
    def show_option(self):
        self.hide_options()
        seg_method = str(self.ui.comboBox_method.currentText())
        for method,frame in self.options:
            if method == seg_method:
                if method == "Baatz" or method == "Region Growing":
                    self.ui.radioButton_floaters.show()
                    self.ui.radioButton_integers.show()
                else:
                    self.ui.radioButton_floaters.hide()
                    self.ui.radioButton_integers.hide()
                '''
                if seg_method == "Meanshift" or seg_method == "Morphological Profiles":
                    self.ui.checkBox_optimizer.setChecked(0)
                    self.ui.checkBox_optimizer.hide()
                else:
                    self.ui.checkBox_optimizer.show()
                '''
                frame.show()
                return
    def setPath_input(self):
        if self.ui.comboBox_input.currentIndex() == 0:
            fileName = QFileDialog.getOpenFileName(self,"Open Image", "~/","Image Files (*.tiff *.tif)");
            if fileName !="":
                self.ui.comboBox_input.setItemText(0, _translate("Pansharp", "[Choose from a file..] "+fileName, None))
            else:
                self.ui.comboBox_input.setItemText(0, _translate("Pansharp", "[Choose from a file..]", None))
    def setPath_output(self):
        fileName = QFileDialog.getSaveFileName(self,"Output Shapefile", "~/","ESRI Shapefile Files (*.shp)");
        if fileName !="":
            self.ui.lineEdit_output.setText(fileName)
    def setPath_optimizer_input(self):
        fileName = QFileDialog.getOpenFileName(self,"Open Shapefile", "~/","ESRI Shapefile Files (*.shp)");
        if fileName !="":
            self.ui.lineEdit_optimizer_input.setText(fileName)
    def select_optimizer_frame(self):
        checked = bool(self.ui.checkBox_optimizer.isChecked())
        seg_method = str(self.ui.comboBox_method.currentText())
        if checked:
            self.ui.groupBox_optimizer.show()
            self.ui.groupBox_options.hide()
        else:
            self.ui.groupBox_optimizer.hide()
            self.ui.groupBox_options.show()

class FeaturesDialog(QtGui.QDialog, Ui_Features):
    def __init__(self):
        QtGui.QDialog.__init__(self)
        self.ui = Ui_Features()
        self.ui.setupUi(self)
        #QObject.connect(self.ui.pushButton_input, SIGNAL("clicked()"), self.setPath_input)
        #QObject.connect(self.ui.pushButton_training, SIGNAL("clicked()"), self.setPath_training)
        QObject.connect(self.ui.comboBox_input, SIGNAL("activated(const QString&)"), self.setPath_input)
        QObject.connect(self.ui.comboBox_training, SIGNAL("activated(const QString&)"), self.setPath_training)
        QObject.connect(self.ui.pushButton_output, SIGNAL("clicked()"), self.setPath_output)
    def setPath_input(self):
        if self.ui.comboBox_input.currentIndex() == 0:
            fileName = QFileDialog.getOpenFileName(self,"Open Image", "~/","Image Files (*.tiff *.tif)");
            if fileName !="":
                self.ui.comboBox_input.setItemText(0, _translate("Pansharp", "[Choose from a file..] "+fileName, None))
            else:
                self.ui.comboBox_input.setItemText(0, _translate("Pansharp", "[Choose from a file..]", None))
    def setPath_training(self):
        if self.ui.comboBox_training.currentIndex() == 0:
            fileName = QFileDialog.getOpenFileName(self,"Input Shapefile", "~/","ESRI Shapefile Files (*.shp)");
            if fileName !="":
                self.ui.comboBox_training.setItemText(0, _translate("Pansharp", "[Choose from a file..] "+fileName, None))
            else:
                self.ui.comboBox_training.setItemText(0, _translate("Pansharp", "[Choose from a file..]", None))
    def setPath_output(self):
        #fileName = QFileDialog.getOpenFileName(self,"Input Shapefile", "~/","ESRI Shapefile Files (*.shp)");
        fileName = QFileDialog.getSaveFileName(self,"Output Shapefile", "~/","ESRI Shapefile Files (*.shp)");
        if fileName !="":
            self.ui.lineEdit_output.setText(fileName)

class BuildHeightDialog(QtGui.QDialog, Ui_BuildHeight):
    def __init__(self):
        QtGui.QDialog.__init__(self)
        self.ui = Ui_BuildHeight()
        self.ui.setupUi(self)
        #QObject.connect(self.ui.pushButton_input_buldings, SIGNAL("clicked()"), self.setPath_input_building)
        #QObject.connect(self.ui.pushButton_input_shadows, SIGNAL("clicked()"), self.setPath_input_shadow)
        QObject.connect(self.ui.comboBox_input_buildings, SIGNAL("activated(const QString&)"), self.setPath_input)
        QObject.connect(self.ui.comboBox_input_shadows, SIGNAL("activated(const QString&)"), self.setPath_input_shadow)
        QObject.connect(self.ui.pushButton_output, SIGNAL("clicked()"), self.setPath_output)
    def setPath_input(self):
        if self.ui.comboBox_input_buildings.currentIndex() == 0:
            fileName = QFileDialog.getOpenFileName(self,"Input Shapefile", "~/","ESRI Shapefile Files (*.shp)");
            if fileName !="":
                self.ui.comboBox_input_buildings.setItemText(0, _translate("Pansharp", "[Choose from a file..] "+fileName, None))
            else:
                self.ui.comboBox_input_buildings.setItemText(0, _translate("Pansharp", "[Choose from a file..]", None))
    def setPath_input_shadow(self):
        if self.ui.comboBox_input_shadows.currentIndex() == 0:
            fileName = QFileDialog.getOpenFileName(self,"Input Shapefile", "~/","ESRI Shapefile Files (*.shp)");
            if fileName !="":
                self.ui.comboBox_input_shadows.setItemText(0, _translate("Pansharp", "[Choose from a file..] "+fileName, None))
            else:
                self.ui.comboBox_input_shadows.setItemText(0, _translate("Pansharp", "[Choose from a file..]", None))
    def setPath_output(self):
        fileName = QFileDialog.getSaveFileName(self,"Output Shapefile", "~/","ESRI Shapefile Files (*.shp)");
        if fileName !="":
            self.ui.lineEdit_output.setText(fileName)

class CoregistrationDialog(QtGui.QDialog, Ui_Coregistration):
    def __init__(self):
        QtGui.QDialog.__init__(self)
        self.ui = Ui_Coregistration()
        self.ui.setupUi(self)
        self.ui.groupBox_clip.show()
        self.ui.groupBox_grid.hide()
        QObject.connect(self.ui.pushButton_reference, SIGNAL("clicked()"), self.setPath_reference)
        QObject.connect(self.ui.pushButton_tobechange, SIGNAL("clicked()"), self.setPath_tobechange)
        QObject.connect(self.ui.comboBox_input_shape, SIGNAL("activated(const QString&)"), self.setPath_input_shapefile)
        QObject.connect(self.ui.comboBox_select_crop, SIGNAL("currentIndexChanged(const QString&)"), self.select_options_frame)
    def setPath_reference(self):
        fileName = QFileDialog.getExistingDirectory(self,"Select Folder", "~");
        if fileName !="":
            self.ui.lineEdit_reference.setText(fileName)
    def setPath_tobechange(self):
        fileName = QFileDialog.getExistingDirectory(self,"Select Folder", "~");
        if fileName !="":
            self.ui.lineEdit_tobechange.setText(fileName)
    def setPath_input_shapefile(self):
        if self.ui.comboBox_input_shape.currentIndex() == 0:
            fileName = QFileDialog.getOpenFileName(self,"Input Shapefile", "~/","ESRI Shapefile Files (*.shp)");
            if fileName !="":
                self.ui.comboBox_input_shape.setItemText(0, _translate("Pansharp", "[Choose from a file..] "+fileName, None))
            else:
                self.ui.comboBox_input_shape.setItemText(0, _translate("Pansharp", "[Choose from a file..]", None))
    def select_options_frame(self):
        option = str(self.ui.comboBox_select_crop.currentText())
        if option == "Clip":
            self.ui.groupBox_clip.show()
            self.ui.groupBox_grid.hide()
        else:
            self.ui.groupBox_clip.hide()
            self.ui.groupBox_grid.show()

class FootprintsDialog(QtGui.QDialog, Ui_Footprints):
    def __init__(self):
        QtGui.QDialog.__init__(self)
        self.ui = Ui_Footprints()
        self.ui.setupUi(self)
        #QObject.connect(self.ui.pushButton_pansharp, SIGNAL("clicked()"), self.setPath_pansharp)
        #QObject.connect(self.ui.pushButton_training, SIGNAL("clicked()"), self.setPath_training)
        QObject.connect(self.ui.comboBox_pansharp, SIGNAL("activated(const QString&)"), self.setPath_pansharp)
        QObject.connect(self.ui.comboBox_training, SIGNAL("activated(const QString&)"), self.setPath_training)
        QObject.connect(self.ui.pushButton_output, SIGNAL("clicked()"), self.setPath_output)
        QObject.connect(self.ui.pushButton_add, SIGNAL("clicked()"), self.addItem_classes)
        QObject.connect(self.ui.pushButton_clear, SIGNAL("clicked()"), self.clear_selected)
        QObject.connect(self.ui.pushButton_clear_all, SIGNAL("clicked()"), self.clear_all)
    def setPath_pansharp(self):
        if self.ui.comboBox_pansharp.currentIndex() == 0:
            fileName = QFileDialog.getOpenFileName(self,"Open Image", "~/","Image Files (*.tiff *.tif)");
            if fileName !="":
                self.ui.comboBox_pansharp.setItemText(0, _translate("Pansharp", "[Choose from a file..] "+fileName, None))
            else:
                self.ui.comboBox_pansharp.setItemText(0, _translate("Pansharp", "[Choose from a file..]", None))
    def setPath_training(self):
        if self.ui.comboBox_training.currentIndex() == 0:
            fileName = QFileDialog.getOpenFileName(self,"Input Shapefile", "~/","ESRI Shapefile Files (*.shp)");
            if fileName !="":
                self.ui.comboBox_training.setItemText(0, _translate("Pansharp", "[Choose from a file..] "+fileName, None))
            else:
                self.ui.comboBox_training.setItemText(0, _translate("Pansharp", "[Choose from a file..]", None))
    def setPath_output(self):
        fileName = QFileDialog.getSaveFileName(self,"Output Shapefile", "~/","ESRI Shapefile Files (*.shp)");
        if fileName !="":
            self.ui.lineEdit_output.setText(fileName)
    def addItem_classes(self):
        value = str(self.ui.lineEdit_classes.text())
        self.ui.listWidget.addItem(value)
        self.ui.lineEdit_classes.clear()
    def clear_selected(self):
        for SelectedItem in self.ui.listWidget.selectedItems():
            self.ui.listWidget.takeItem(self.ui.listWidget.row(SelectedItem))
    def clear_all(self):
        self.ui.listWidget.clear()
        
class StackSatelliteDialog(QtGui.QDialog, Ui_StackSatellite):
    def __init__(self):
        QtGui.QDialog.__init__(self)
        self.ui = Ui_StackSatellite()
        self.ui.setupUi(self)
        QObject.connect(self.ui.pushButton_satellite_folder, SIGNAL("clicked()"), self.setPath_satellite_folder)
        QObject.connect(self.ui.comboBox_input_shapefile, SIGNAL("activated(const QString&)"), self.setPath_inputshapefile)
        QObject.connect(self.ui.pushButton_reference_directory, SIGNAL("clicked()"), self.setPath_reference_directory)
        QObject.connect(self.ui.checkBox_reference_diretory, SIGNAL("stateChanged(int)"), self.reference_directory)
        QObject.connect(self.ui.checkBox_restrict_to_city, SIGNAL("stateChanged(int)"), self.restrict_to_city)
    def setPath_satellite_folder(self):
        fileName = QFileDialog.getExistingDirectory(self,"Select Folder", "~");
        if fileName !="":
            self.ui.lineEdit_satellite_folder.setText(fileName)
    def setPath_inputshapefile(self):
        if self.ui.comboBox_input_shapefile.currentIndex() == 0:
            fileName = QFileDialog.getOpenFileName(self,"Input Shapefile", "~/","ESRI Shapefile Files (*.shp)");
            if fileName !="":
                self.ui.comboBox_input_shapefile.setItemText(0, _translate("Shapefile", "[Choose from a file..] "+fileName, None))
            else:
                self.ui.comboBox_input_shapefile.setItemText(0, _translate("Shapefile", "[Choose from a file..]", None))
    def setPath_reference_directory(self):
        fileName = QFileDialog.getExistingDirectory(self,"Select Folder", "~");
        if fileName !="":
            self.ui.lineEdit_reference_directory.setText(fileName)
    def reference_directory(self):
        checked = bool(self.ui.checkBox_reference_diretory.isChecked())
        if checked:
            self.ui.lineEdit_reference_directory.setEnabled(1)
            self.ui.pushButton_reference_directory.setEnabled(1)
        else:
            self.ui.lineEdit_reference_directory.setEnabled(0)
            self.ui.pushButton_reference_directory.setEnabled(0) 
    def restrict_to_city(self):
        checked = bool(self.ui.checkBox_restrict_to_city.isChecked())
        if checked: self.ui.comboBox_input_shapefile.setEnabled(1)
        else: self.ui.comboBox_input_shapefile.setEnabled(0)

class DensityDialog(QtGui.QDialog, Ui_Density):
    def __init__(self):
        QtGui.QDialog.__init__(self)
        self.ui = Ui_Density()
        self.ui.setupUi(self)
        #QObject.connect(self.ui.pushButton_building_shape, SIGNAL("clicked()"), self.setPath_building_shape)
        QObject.connect(self.ui.comboBox_building_shape, SIGNAL("activated(const QString&)"), self.setPath_building_shape)
        QObject.connect(self.ui.pushButton_output_shapefile, SIGNAL("clicked()"), self.setPath_output_shapefile)

    def setPath_building_shape(self):
        if self.ui.comboBox_building_shape.currentIndex() == 0:
            fileName = QFileDialog.getOpenFileName(self,"Output Shapefile", "~/","ESRI Shapefile Files (*.shp)");
            if fileName !="":
                self.ui.comboBox_building_shape.setItemText(0, _translate("Pansharp", "[Choose from a file..] "+fileName, None))
            else:
                self.ui.comboBox_building_shape.setItemText(0, _translate("Pansharp", "[Choose from a file..]", None))
    def setPath_output_shapefile(self):
        fileName = QFileDialog.getSaveFileName(self,"Output Shapefile", "~/","ESRI Shapefile Files (*.shp)");
        if fileName !="":
            self.ui.lineEdit_output_shapefile.setText(fileName)

class TemporalDialog(QtGui.QDialog, Ui_Temporal):
    def __init__(self):
        QtGui.QDialog.__init__(self)
        self.ui = Ui_Temporal()
        self.ui.setupUi(self)
        QObject.connect(self.ui.pushButton_folder, SIGNAL("clicked()"), self.setPath_input_folder)
        QObject.connect(self.ui.comboBox_mask, SIGNAL("activated(const QString&)"), self.setPath_inputmask)

    def setPath_input_folder(self):
        fileName = QFileDialog.getExistingDirectory(self,"Select Folder", "~");
        if fileName !="":
            self.ui.lineEdit_folder.setText(fileName)

    def setPath_inputmask(self):
        if self.ui.comboBox_mask.currentIndex() == 0:
            fileName = QFileDialog.getOpenFileName(self,"Input Shapefile", "~/","ESRI Shapefile Files (*.shp)");
            if fileName !="":
                self.ui.comboBox_mask.setItemText(0, _translate("Pansharp", "[Choose from a file..] "+fileName, None))
            else:
                self.ui.comboBox_mask.setItemText(0, _translate("Pansharp", "[Choose from a file..]", None))

