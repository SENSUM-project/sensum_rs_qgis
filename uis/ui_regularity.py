# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_regularity.ui'
#
# Created: Mon Oct 20 18:44:39 2014
#      by: PyQt4 UI code generator 4.10.2
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

class Ui_Regularity(object):
    def setupUi(self, Regularity):
        Regularity.setObjectName(_fromUtf8("Regularity"))
        Regularity.resize(467, 365)
        Regularity.setMinimumSize(QtCore.QSize(467, 365))
        Regularity.setMaximumSize(QtCore.QSize(467, 365))
        self.widget = QtGui.QWidget(Regularity)
        self.widget.setGeometry(QtCore.QRect(20, 140, 441, 207))
        self.widget.setObjectName(_fromUtf8("widget"))
        self.buttonBox = QtGui.QDialogButtonBox(self.widget)
        self.buttonBox.setGeometry(QtCore.QRect(9, 140, 160, 27))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.label_building_shape = QtGui.QLabel(self.widget)
        self.label_building_shape.setGeometry(QtCore.QRect(9, 9, 261, 16))
        self.label_building_shape.setObjectName(_fromUtf8("label_building_shape"))
        self.label_output_shapefile = QtGui.QLabel(self.widget)
        self.label_output_shapefile.setGeometry(QtCore.QRect(10, 70, 111, 16))
        self.label_output_shapefile.setObjectName(_fromUtf8("label_output_shapefile"))
        self.lineEdit_output_shapefile = QtGui.QLineEdit(self.widget)
        self.lineEdit_output_shapefile.setGeometry(QtCore.QRect(10, 90, 351, 21))
        self.lineEdit_output_shapefile.setObjectName(_fromUtf8("lineEdit_output_shapefile"))
        self.pushButton_output_shapefile = QtGui.QPushButton(self.widget)
        self.pushButton_output_shapefile.setGeometry(QtCore.QRect(380, 90, 41, 21))
        self.pushButton_output_shapefile.setObjectName(_fromUtf8("pushButton_output_shapefile"))
        self.comboBox_building_shape = QtGui.QComboBox(self.widget)
        self.comboBox_building_shape.setGeometry(QtCore.QRect(10, 30, 401, 20))
        self.comboBox_building_shape.setObjectName(_fromUtf8("comboBox_building_shape"))
        self.comboBox_building_shape.addItem(_fromUtf8(""))
        self.label_title = QtGui.QLabel(Regularity)
        self.label_title.setGeometry(QtCore.QRect(160, 100, 131, 21))
        self.label_title.setObjectName(_fromUtf8("label_title"))
        self.logo_sensum = QtGui.QLabel(Regularity)
        self.logo_sensum.setGeometry(QtCore.QRect(10, 20, 241, 61))
        self.logo_sensum.setText(_fromUtf8(""))
        self.logo_sensum.setPixmap(QtGui.QPixmap(_fromUtf8(".sensum/sensum.png")))
        self.logo_sensum.setObjectName(_fromUtf8("logo_sensum"))
        self.logo_unipv = QtGui.QLabel(Regularity)
        self.logo_unipv.setGeometry(QtCore.QRect(350, 10, 91, 81))
        self.logo_unipv.setText(_fromUtf8(""))
        self.logo_unipv.setPixmap(QtGui.QPixmap(_fromUtf8(".sensum/unipv.png")))
        self.logo_unipv.setObjectName(_fromUtf8("logo_unipv"))
        self.logo_eucentre = QtGui.QLabel(Regularity)
        self.logo_eucentre.setGeometry(QtCore.QRect(270, 10, 71, 81))
        self.logo_eucentre.setText(_fromUtf8(""))
        self.logo_eucentre.setPixmap(QtGui.QPixmap(_fromUtf8(".sensum/eucentre.png")))
        self.logo_eucentre.setObjectName(_fromUtf8("logo_eucentre"))

        self.retranslateUi(Regularity)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("accepted()")), Regularity.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("rejected()")), Regularity.reject)
        QtCore.QMetaObject.connectSlotsByName(Regularity)

    def retranslateUi(self, Regularity):
        Regularity.setWindowTitle(_translate("Regularity", "SensumTools", None))
        self.label_building_shape.setText(_translate("Regularity", "Shapefile of the Buildings", None))
        self.label_output_shapefile.setText(_translate("Regularity", "Output Shapefile", None))
        self.pushButton_output_shapefile.setText(_translate("Regularity", "...", None))
        self.comboBox_building_shape.setItemText(0, _translate("Regularity", "[Choose from a file..]", "Click to select File"))
        self.label_title.setText(_translate("Regularity", "<html><head/><body><p><span style=\" font-size:16pt;\">REGULARITY</span></p></body></html>", None))

