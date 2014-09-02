# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'uis/ui_pansharp.ui'
#
# Created: Wed Aug 27 10:14:27 2014
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

class Ui_Pansharp(object):
    def setupUi(self, Pansharp):
        Pansharp.setObjectName(_fromUtf8("Pansharp"))
        Pansharp.resize(467, 365)
        Pansharp.setMinimumSize(QtCore.QSize(467, 365))
        Pansharp.setMaximumSize(QtCore.QSize(467, 365))
        self.widget = QtGui.QWidget(Pansharp)
        self.widget.setGeometry(QtCore.QRect(20, 140, 441, 207))
        self.widget.setObjectName(_fromUtf8("widget"))
        self.lineEdit_output = QtGui.QLineEdit(self.widget)
        self.lineEdit_output.setGeometry(QtCore.QRect(9, 141, 321, 21))
        self.lineEdit_output.setObjectName(_fromUtf8("lineEdit_output"))
        self.pushButton_output = QtGui.QPushButton(self.widget)
        self.pushButton_output.setGeometry(QtCore.QRect(355, 138, 77, 27))
        self.pushButton_output.setObjectName(_fromUtf8("pushButton_output"))
        self.buttonBox = QtGui.QDialogButtonBox(self.widget)
        self.buttonBox.setGeometry(QtCore.QRect(9, 171, 160, 27))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.label_panchromatic = QtGui.QLabel(self.widget)
        self.label_panchromatic.setGeometry(QtCore.QRect(9, 63, 123, 16))
        self.label_panchromatic.setObjectName(_fromUtf8("label_panchromatic"))
        self.label_multiband = QtGui.QLabel(self.widget)
        self.label_multiband.setGeometry(QtCore.QRect(9, 9, 101, 16))
        self.label_multiband.setObjectName(_fromUtf8("label_multiband"))
        self.label_output = QtGui.QLabel(self.widget)
        self.label_output.setGeometry(QtCore.QRect(9, 117, 83, 16))
        self.label_output.setObjectName(_fromUtf8("label_output"))
        self.comboBox_multiband = QtGui.QComboBox(self.widget)
        self.comboBox_multiband.setGeometry(QtCore.QRect(10, 30, 411, 20))
        self.comboBox_multiband.setObjectName(_fromUtf8("comboBox_multiband"))
        self.comboBox_multiband.addItem(_fromUtf8(""))
        self.comboBox_panchromatic = QtGui.QComboBox(self.widget)
        self.comboBox_panchromatic.setGeometry(QtCore.QRect(10, 90, 411, 20))
        self.comboBox_panchromatic.setObjectName(_fromUtf8("comboBox_panchromatic"))
        self.comboBox_panchromatic.addItem(_fromUtf8(""))
        self.label_title = QtGui.QLabel(Pansharp)
        self.label_title.setGeometry(QtCore.QRect(140, 100, 181, 21))
        self.label_title.setObjectName(_fromUtf8("label_title"))
        self.logo_sensum = QtGui.QLabel(Pansharp)
        self.logo_sensum.setGeometry(QtCore.QRect(10, 20, 241, 61))
        self.logo_sensum.setText(_fromUtf8(""))
        self.logo_sensum.setPixmap(QtGui.QPixmap(_fromUtf8(".sensum/sensum.png")))
        self.logo_sensum.setObjectName(_fromUtf8("logo_sensum"))
        self.logo_unipv = QtGui.QLabel(Pansharp)
        self.logo_unipv.setGeometry(QtCore.QRect(350, 10, 91, 81))
        self.logo_unipv.setText(_fromUtf8(""))
        self.logo_unipv.setPixmap(QtGui.QPixmap(_fromUtf8(".sensum/unipv.png")))
        self.logo_unipv.setObjectName(_fromUtf8("logo_unipv"))
        self.logo_eucentre = QtGui.QLabel(Pansharp)
        self.logo_eucentre.setGeometry(QtCore.QRect(270, 10, 71, 81))
        self.logo_eucentre.setText(_fromUtf8(""))
        self.logo_eucentre.setPixmap(QtGui.QPixmap(_fromUtf8(".sensum/eucentre.png")))
        self.logo_eucentre.setObjectName(_fromUtf8("logo_eucentre"))

        self.retranslateUi(Pansharp)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("accepted()")), Pansharp.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("rejected()")), Pansharp.reject)
        QtCore.QMetaObject.connectSlotsByName(Pansharp)

    def retranslateUi(self, Pansharp):
        Pansharp.setWindowTitle(_translate("Pansharp", "SensumTools", None))
        self.pushButton_output.setText(_translate("Pansharp", "...", None))
        self.label_panchromatic.setText(_translate("Pansharp", "Panchromatic Image", None))
        self.label_multiband.setText(_translate("Pansharp", "Multiband Image", None))
        self.label_output.setText(_translate("Pansharp", "Output Image", None))
        self.comboBox_multiband.setItemText(0, _translate("Pansharp", "[Choose from a file..]", "Click to select File"))
        self.comboBox_panchromatic.setItemText(0, _translate("Pansharp", "[Choose from a file..]", "Click to select File"))
        self.label_title.setText(_translate("Pansharp", "<html><head/><body><p><span style=\" font-size:16pt;\">PANSHARPENING</span></p></body></html>", None))

