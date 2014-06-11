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
                
        QObject.connect(self.ui.pushButton_multiband, SIGNAL("clicked()"), self.setPath_multiband)
        QObject.connect(self.ui.pushButton_panchromatic, SIGNAL("clicked()"), self.setPath_panchromatic)
        QObject.connect(self.ui.pushButton_output, SIGNAL("clicked()"), self.setPath_output)

    def setPath_multiband(self):
        fileName = QFileDialog.getOpenFileName(self,"Open Image", "~/","Image Files (*.tiff *.tif)");
        if fileName !="":
            self.ui.lineEdit_multiband.setText(fileName)
    def setPath_panchromatic(self):
        fileName = QFileDialog.getOpenFileName(self,"Open Image", "~/","Image Files (*.tif)");
        if fileName !="":
            self.ui.lineEdit_panchromatic.setText(fileName)
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
        QObject.connect(self.ui.pushButton_input, SIGNAL("clicked()"), self.setPath_input)
        QObject.connect(self.ui.pushButton_output, SIGNAL("clicked()"), self.setPath_output)
        QObject.connect(self.ui.pushButton_training, SIGNAL("clicked()"), self.setPath_training)
        QObject.connect(self.ui.comboBox_supervised, SIGNAL("currentIndexChanged(const QString&)"), self.select_options_frame)
    def setPath_input(self):
        fileName = QFileDialog.getOpenFileName(self,"Open Image", "~/","Image Files (*.tiff *.tif)");
        if fileName !="":
            self.ui.lineEdit_input.setText(fileName)
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
        QObject.connect(self.ui.pushButton_input, SIGNAL("clicked()"), self.setPath_input)
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
        fileName = QFileDialog.getOpenFileName(self,"Open Image", "~/","Image Files (*.tiff *.tif)");
        if fileName !="":
            self.ui.lineEdit_input.setText(fileName)
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
        QObject.connect(self.ui.pushButton_input, SIGNAL("clicked()"), self.setPath_input)
        QObject.connect(self.ui.pushButton_output, SIGNAL("clicked()"), self.setPath_output)
        QObject.connect(self.ui.pushButton_training, SIGNAL("clicked()"), self.setPath_training)
    def setPath_input(self):
        fileName = QFileDialog.getOpenFileName(self,"Open Image", "~/","Image Files (*.tiff *.tif)");
        if fileName !="":
            self.ui.lineEdit_input.setText(fileName)
    def setPath_output(self):
        fileName = QFileDialog.getSaveFileName(self,"Output Image", "~/","Image Files (*.shp)");
        if fileName !="":
            self.ui.lineEdit_output.setText(fileName)
    def setPath_training(self):
        fileName = QFileDialog.getOpenFileName(self,"Input Shapefile", "~/","ESRI Shapefile Files (*.shp)");
        if fileName !="":
            self.ui.lineEdit_training.setText(fileName)

class BuildHeightDialog(QtGui.QDialog, Ui_BuildHeight):
    def __init__(self):
        QtGui.QDialog.__init__(self)
        self.ui = Ui_BuildHeight()
        self.ui.setupUi(self)
        QObject.connect(self.ui.pushButton_input_buldings, SIGNAL("clicked()"), self.setPath_input_building)
        QObject.connect(self.ui.pushButton_input_shadows, SIGNAL("clicked()"), self.setPath_input_shadow)
        QObject.connect(self.ui.pushButton_output, SIGNAL("clicked()"), self.setPath_output)
    def setPath_input_building(self):
        fileName = QFileDialog.getOpenFileName(self,"Input Shapefile", "~/","ESRI Shapefile Files (*.shp)");
        if fileName !="":
            self.ui.lineEdit_input_buildings.setText(fileName)
    def setPath_input_shadow(self):
        fileName = QFileDialog.getOpenFileName(self,"Input Shapefile", "~/","ESRI Shapefile Files (*.shp)");
        if fileName !="":
            self.ui.lineEdit_input_shadows.setText(fileName)
    def setPath_output(self):
        fileName = QFileDialog.getSaveFileName(self,"Output Shapefile", "~/","ESRI Shapefile Files (*.shp)");
        if fileName !="":
            self.ui.lineEdit_output.setText(fileName)

class CoregistrationDialog(QtGui.QDialog, Ui_Coregistration):
    def __init__(self):
        QtGui.QDialog.__init__(self)
        self.ui = Ui_Coregistration()
        self.ui.setupUi(self)
        QObject.connect(self.ui.pushButton_reference, SIGNAL("clicked()"), self.setPath_reference)
        QObject.connect(self.ui.pushButton_tobechange, SIGNAL("clicked()"), self.setPath_tobechange)
    def setPath_reference(self):
        fileName = QFileDialog.getExistingDirectory(self,"Select Folder", "~");
        if fileName !="":
            self.ui.lineEdit_reference.setText(fileName)
    def setPath_tobechange(self):
        fileName = QFileDialog.getExistingDirectory(self,"Select Folder", "~");
        if fileName !="":
            self.ui.lineEdit_tobechange.setText(fileName)

class FootprintsDialog(QtGui.QDialog, Ui_Footprints):
    def __init__(self):
        QtGui.QDialog.__init__(self)
        self.ui = Ui_Footprints()
        self.ui.setupUi(self)
        QObject.connect(self.ui.pushButton_pansharp, SIGNAL("clicked()"), self.setPath_pansharp)
        QObject.connect(self.ui.pushButton_training, SIGNAL("clicked()"), self.setPath_training)
        QObject.connect(self.ui.pushButton_add, SIGNAL("clicked()"), self.addItem_classes)
        QObject.connect(self.ui.pushButton_clear, SIGNAL("clicked()"), self.clear_selected)
        QObject.connect(self.ui.pushButton_clear_all, SIGNAL("clicked()"), self.clear_all)
    def setPath_pansharp(self):
        fileName = QFileDialog.getOpenFileName(self,"Open Image", "~/","Image Files (*.tiff *.tif)");
        if fileName !="":
            self.ui.lineEdit_pansharp.setText(fileName)
    def setPath_training(self):
        fileName = QFileDialog.getOpenFileName(self,"Input Shapefile", "~/","ESRI Shapefile Files (*.shp)");
        if fileName !="":
            self.ui.lineEdit_training.setText(fileName)
    def addItem_classes(self):
        value = str(self.ui.lineEdit_classes.text())
        self.ui.listWidget.addItem(value)
        self.ui.lineEdit_classes.clear()
    def clear_selected(self):
        for SelectedItem in self.ui.listWidget.selectedItems():
            self.ui.listWidget.takeItem(self.ui.listWidget.row(SelectedItem))
    def clear_all(self):
        self.ui.listWidget.clear()