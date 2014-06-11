# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'uis/ui_pansharp.ui'
#
# Created: Wed Jun 11 12:14:53 2014
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
        Pansharp.resize(316, 272)
        Pansharp.setMinimumSize(QtCore.QSize(316, 272))
        Pansharp.setMaximumSize(QtCore.QSize(316, 272))
        self.widget = QtGui.QWidget(Pansharp)
        self.widget.setGeometry(QtCore.QRect(10, 50, 291, 207))
        self.widget.setObjectName(_fromUtf8("widget"))
        self.gridLayout = QtGui.QGridLayout(self.widget)
        self.gridLayout.setMargin(0)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.lineEdit_multiband = QtGui.QLineEdit(self.widget)
        self.lineEdit_multiband.setObjectName(_fromUtf8("lineEdit_multiband"))
        self.gridLayout.addWidget(self.lineEdit_multiband, 1, 0, 1, 1)
        self.pushButton_multiband = QtGui.QPushButton(self.widget)
        self.pushButton_multiband.setObjectName(_fromUtf8("pushButton_multiband"))
        self.gridLayout.addWidget(self.pushButton_multiband, 1, 1, 1, 1)
        self.lineEdit_panchromatic = QtGui.QLineEdit(self.widget)
        self.lineEdit_panchromatic.setObjectName(_fromUtf8("lineEdit_panchromatic"))
        self.gridLayout.addWidget(self.lineEdit_panchromatic, 3, 0, 1, 1)
        self.lineEdit_output = QtGui.QLineEdit(self.widget)
        self.lineEdit_output.setObjectName(_fromUtf8("lineEdit_output"))
        self.gridLayout.addWidget(self.lineEdit_output, 5, 0, 1, 1)
        self.pushButton_output = QtGui.QPushButton(self.widget)
        self.pushButton_output.setObjectName(_fromUtf8("pushButton_output"))
        self.gridLayout.addWidget(self.pushButton_output, 5, 1, 1, 1)
        self.buttonBox = QtGui.QDialogButtonBox(self.widget)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.gridLayout.addWidget(self.buttonBox, 6, 0, 1, 2)
        self.label_panchromatic = QtGui.QLabel(self.widget)
        self.label_panchromatic.setObjectName(_fromUtf8("label_panchromatic"))
        self.gridLayout.addWidget(self.label_panchromatic, 2, 0, 1, 2)
        self.label_multiband = QtGui.QLabel(self.widget)
        self.label_multiband.setObjectName(_fromUtf8("label_multiband"))
        self.gridLayout.addWidget(self.label_multiband, 0, 0, 1, 2)
        self.label_output = QtGui.QLabel(self.widget)
        self.label_output.setObjectName(_fromUtf8("label_output"))
        self.gridLayout.addWidget(self.label_output, 4, 0, 1, 2)
        self.pushButton_panchromatic = QtGui.QPushButton(self.widget)
        self.pushButton_panchromatic.setObjectName(_fromUtf8("pushButton_panchromatic"))
        self.gridLayout.addWidget(self.pushButton_panchromatic, 3, 1, 1, 1)
        self.label_title = QtGui.QLabel(Pansharp)
        self.label_title.setGeometry(QtCore.QRect(100, 20, 121, 21))
        self.label_title.setObjectName(_fromUtf8("label_title"))

        self.retranslateUi(Pansharp)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("accepted()")), Pansharp.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("rejected()")), Pansharp.reject)
        QtCore.QMetaObject.connectSlotsByName(Pansharp)

    def retranslateUi(self, Pansharp):
        Pansharp.setWindowTitle(_translate("Pansharp", "Sensum", None))
        self.pushButton_multiband.setText(_translate("Pansharp", "...", None))
        self.pushButton_output.setText(_translate("Pansharp", "...", None))
        self.label_panchromatic.setText(_translate("Pansharp", "Panchromatic Image", None))
        self.label_multiband.setText(_translate("Pansharp", "Multiband Image", None))
        self.label_output.setText(_translate("Pansharp", "Output Image", None))
        self.pushButton_panchromatic.setText(_translate("Pansharp", "...", None))
        self.label_title.setText(_translate("Pansharp", "<html><head/><body><p><span style=\" font-size:16pt;\">PANSHARP</span></p></body></html>", None))

