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
from uis.ui_features import Ui_Features
from uis.ui_build_height import Ui_BuildHeight
from uis.ui_coregistration import Ui_Coregistration
from uis.ui_footprints import Ui_Footprints
from uis.ui_stacksatellite import Ui_StackSatellite
from uis.ui_density import Ui_Density
from uis.ui_change_detection import Ui_ChangeDetection
from uis.ui_regularity import Ui_Regularity

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
        self.ui = Ui_Progress()
        self.ui.setupUi(self)
        self.ui.progressBar.setValue( 0 )

class FeaturesDialog(QtGui.QDialog, Ui_Features):
    def __init__(self):
        QtGui.QDialog.__init__(self)
        self.ui = Ui_Features()
        self.ui.setupUi(self)
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
        fileName = QFileDialog.getSaveFileName(self,"Output Shapefile", "~/","ESRI Shapefile Files (*.shp)");
        if fileName !="":
            self.ui.lineEdit_output.setText(fileName)

class BuildHeightDialog(QtGui.QDialog, Ui_BuildHeight):
    def __init__(self):
        QtGui.QDialog.__init__(self)
        self.ui = Ui_BuildHeight()
        self.ui.setupUi(self)
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
        self.ui.groupBox_unsupervised_classification.hide()
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
            self.ui.groupBox_unsupervised_classification.hide()
        elif option == "Grid":
            self.ui.groupBox_clip.hide()
            self.ui.groupBox_grid.show()
            self.ui.groupBox_unsupervised_classification.hide()
        elif option == "Unsupervised Classification":
            self.ui.groupBox_clip.hide()
            self.ui.groupBox_grid.hide()
            self.ui.groupBox_unsupervised_classification.show()

class FootprintsDialog(QtGui.QDialog, Ui_Footprints):
    def __init__(self):
        QtGui.QDialog.__init__(self)
        self.ui = Ui_Footprints()
        self.ui.setupUi(self)
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
        self.ui.lineEdit_reference_directory.setEnabled(0)
        self.ui.pushButton_reference_directory.setEnabled(0)
        self.ui.lineEdit_reference_directory.hide()
        self.ui.pushButton_reference_directory.hide()
        self.ui.pushButton_reference_directory.hide()
        self.ui.checkBox_reference_diretory.hide()
        self.options = [("Edison",self.ui.groupBox_edison),("Meanshift",self.ui.groupBox_meanshift)]
        self.show_option()
        QObject.connect(self.ui.pushButton_satellite_folder, SIGNAL("clicked()"), self.setPath_satellite_folder)
        QObject.connect(self.ui.comboBox_input_shapefile, SIGNAL("activated(const QString&)"), self.setPath_inputshapefile)
        QObject.connect(self.ui.pushButton_reference_directory, SIGNAL("clicked()"), self.setPath_reference_directory)
        QObject.connect(self.ui.checkBox_reference_diretory, SIGNAL("stateChanged(int)"), self.reference_directory)
        QObject.connect(self.ui.checkBox_restrict_to_city, SIGNAL("stateChanged(int)"), self.restrict_to_city)
        QObject.connect(self.ui.comboBox_segmentation, SIGNAL("currentIndexChanged(const QString&)"), self.show_option)
    def hide_options(self):
        for method,frame in self.options:
            frame.hide()
    def show_option(self):
        self.hide_options()
        seg_method = str(self.ui.comboBox_segmentation.currentText())
        for method,frame in self.options:
            if method == seg_method:
                frame.show()
                return
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

class RegularityDialog(QtGui.QDialog, Ui_Regularity):
    def __init__(self):
        QtGui.QDialog.__init__(self)
        self.ui = Ui_Regularity()
        self.ui.setupUi(self)
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

class ChangeDialog(QtGui.QDialog, Ui_ChangeDetection):
    def __init__(self):
        QtGui.QDialog.__init__(self)
        self.ui = Ui_ChangeDetection()
        self.ui.setupUi(self)
        QObject.connect(self.ui.pushButton_tobechange, SIGNAL("clicked()"), self.setPath_tobechange)
    def setPath_tobechange(self):
        fileName = QFileDialog.getExistingDirectory(self,"Select Folder", "~");
        if fileName !="":
            self.ui.lineEdit_tobechange.setText(fileName)
