# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'uis/ui_density.ui'
#
# Created: Wed Jun 25 14:25:37 2014
#      by: PyQt4 UI code generator 4.10.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_Density(object):
    def setupUi(self, Density):
        Density.setObjectName(_fromUtf8("Density"))
        Density.resize(467, 365)
        Density.setMinimumSize(QtCore.QSize(467, 365))
        Density.setMaximumSize(QtCore.QSize(467, 365))
        self.widget = QtGui.QWidget(Density)
        self.widget.setGeometry(QtCore.QRect(20, 140, 441, 207))
        self.widget.setObjectName(_fromUtf8("widget"))
        self.lineEdit_building_shape = QtGui.QLineEdit(self.widget)
        self.lineEdit_building_shape.setGeometry(QtCore.QRect(10, 30, 351, 21))
        self.lineEdit_building_shape.setObjectName(_fromUtf8("lineEdit_building_shape"))
        self.pushButton_building_shape = QtGui.QPushButton(self.widget)
        self.pushButton_building_shape.setGeometry(QtCore.QRect(376, 30, 41, 21))
        self.pushButton_building_shape.setObjectName(_fromUtf8("pushButton_building_shape"))
        self.buttonBox = QtGui.QDialogButtonBox(self.widget)
        self.buttonBox.setGeometry(QtCore.QRect(9, 171, 160, 27))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.label_building_shape = QtGui.QLabel(self.widget)
        self.label_building_shape.setGeometry(QtCore.QRect(9, 9, 103, 16))
        self.label_building_shape.setObjectName(_fromUtf8("label_building_shape"))
        self.label_output_shapefile = QtGui.QLabel(self.widget)
        self.label_output_shapefile.setGeometry(QtCore.QRect(10, 120, 111, 16))
        self.label_output_shapefile.setObjectName(_fromUtf8("label_output_shapefile"))
        self.lineEdit_output_shapefile = QtGui.QLineEdit(self.widget)
        self.lineEdit_output_shapefile.setGeometry(QtCore.QRect(10, 140, 351, 21))
        self.lineEdit_output_shapefile.setObjectName(_fromUtf8("lineEdit_output_shapefile"))
        self.pushButton_output_shapefile = QtGui.QPushButton(self.widget)
        self.pushButton_output_shapefile.setGeometry(QtCore.QRect(380, 140, 41, 21))
        self.pushButton_output_shapefile.setObjectName(_fromUtf8("pushButton_output_shapefile"))
        self.doubleSpinBox_radius = QtGui.QDoubleSpinBox(self.widget)
        self.doubleSpinBox_radius.setGeometry(QtCore.QRect(10, 90, 62, 22))
        self.doubleSpinBox_radius.setObjectName(_fromUtf8("doubleSpinBox_radius"))
        self.label_radius = QtGui.QLabel(self.widget)
        self.label_radius.setGeometry(QtCore.QRect(10, 70, 211, 16))
        self.label_radius.setObjectName(_fromUtf8("label_radius"))
        self.label_title = QtGui.QLabel(Density)
        self.label_title.setGeometry(QtCore.QRect(180, 100, 101, 21))
        self.label_title.setObjectName(_fromUtf8("label_title"))
        self.logo_sensum = QtGui.QLabel(Density)
        self.logo_sensum.setGeometry(QtCore.QRect(10, 20, 241, 61))
        self.logo_sensum.setText(_fromUtf8(""))
        self.logo_sensum.setPixmap(QtGui.QPixmap(_fromUtf8("sensum.png")))
        self.logo_sensum.setObjectName(_fromUtf8("logo_sensum"))
        self.logo_unipv = QtGui.QLabel(Density)
        self.logo_unipv.setGeometry(QtCore.QRect(350, 10, 91, 81))
        self.logo_unipv.setText(_fromUtf8(""))
        self.logo_unipv.setPixmap(QtGui.QPixmap(_fromUtf8("unipv.png")))
        self.logo_unipv.setObjectName(_fromUtf8("logo_unipv"))
        self.logo_eucentre = QtGui.QLabel(Density)
        self.logo_eucentre.setGeometry(QtCore.QRect(270, 10, 71, 81))
        self.logo_eucentre.setText(_fromUtf8(""))
        self.logo_eucentre.setPixmap(QtGui.QPixmap(_fromUtf8("eucentre.png")))
        self.logo_eucentre.setObjectName(_fromUtf8("logo_eucentre"))

        self.retranslateUi(Density)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("accepted()")), Density.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("rejected()")), Density.reject)
        QtCore.QMetaObject.connectSlotsByName(Density)

    def retranslateUi(self, Density):
        Density.setWindowTitle(_translate("Density", "SensumTools", None))
        self.pushButton_building_shape.setText(_translate("Density", "...", None))
        self.label_building_shape.setText(_translate("Density", "Building Shape", None))
        self.label_output_shapefile.setText(_translate("Density", "Output Shapefile", None))
        self.pushButton_output_shapefile.setText(_translate("Density", "...", None))
        self.label_radius.setText(_translate("Density", "Radius (expressed in coordinates)", None))
        self.label_title.setText(_translate("Density", "<html><head/><body><p><span style=\" font-size:16pt;\">DENSITY</span></p></body></html>", None))

