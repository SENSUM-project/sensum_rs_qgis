# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_temporal.ui'
#
# Created: Mon Sep 08 16:58:18 2014
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

class Ui_Temporal(object):
    def setupUi(self, Temporal):
        Temporal.setObjectName(_fromUtf8("Temporal"))
        Temporal.resize(578, 392)
        self.buttonBox = QtGui.QDialogButtonBox(Temporal)
        self.buttonBox.setGeometry(QtCore.QRect(310, 330, 161, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.logo_sensum = QtGui.QLabel(Temporal)
        self.logo_sensum.setGeometry(QtCore.QRect(70, 20, 241, 61))
        self.logo_sensum.setText(_fromUtf8(""))
        self.logo_sensum.setPixmap(QtGui.QPixmap(_fromUtf8(".sensum/sensum.png")))
        self.logo_sensum.setObjectName(_fromUtf8("logo_sensum"))
        self.lineEdit_folder = QtGui.QLineEdit(Temporal)
        self.lineEdit_folder.setGeometry(QtCore.QRect(70, 171, 321, 21))
        self.lineEdit_folder.setObjectName(_fromUtf8("lineEdit_folder"))
        self.pushButton_folder = QtGui.QPushButton(Temporal)
        self.pushButton_folder.setGeometry(QtCore.QRect(400, 171, 77, 21))
        self.pushButton_folder.setObjectName(_fromUtf8("pushButton_folder"))
        self.label_title = QtGui.QLabel(Temporal)
        self.label_title.setGeometry(QtCore.QRect(40, 101, 501, 31))
        self.label_title.setObjectName(_fromUtf8("label_title"))
        self.label_multiband = QtGui.QLabel(Temporal)
        self.label_multiband.setGeometry(QtCore.QRect(70, 151, 201, 16))
        self.label_multiband.setObjectName(_fromUtf8("label_multiband"))
        self.label_multiband_2 = QtGui.QLabel(Temporal)
        self.label_multiband_2.setGeometry(QtCore.QRect(69, 221, 121, 16))
        self.label_multiband_2.setObjectName(_fromUtf8("label_multiband_2"))
        self.comboBox_mask = QtGui.QComboBox(Temporal)
        self.comboBox_mask.setGeometry(QtCore.QRect(70, 241, 411, 20))
        self.comboBox_mask.setObjectName(_fromUtf8("comboBox_mask"))
        self.comboBox_mask.addItem(_fromUtf8(""))
        self.spinBox_nclass = QtGui.QSpinBox(Temporal)
        self.spinBox_nclass.setGeometry(QtCore.QRect(150, 300, 53, 22))
        self.spinBox_nclass.setProperty("value", 5)
        self.spinBox_nclass.setObjectName(_fromUtf8("spinBox_nclass"))
        self.label_multiband_3 = QtGui.QLabel(Temporal)
        self.label_multiband_3.setGeometry(QtCore.QRect(120, 280, 131, 16))
        self.label_multiband_3.setObjectName(_fromUtf8("label_multiband_3"))
        self.logo_unipv = QtGui.QLabel(Temporal)
        self.logo_unipv.setGeometry(QtCore.QRect(380, 10, 91, 81))
        self.logo_unipv.setText(_fromUtf8(""))
        self.logo_unipv.setPixmap(QtGui.QPixmap(_fromUtf8(".sensum/cambridge.png")))
        self.logo_unipv.setObjectName(_fromUtf8("logo_unipv"))
        self.pushButton_plot = QtGui.QPushButton(Temporal)
        self.pushButton_plot.setGeometry(QtCore.QRect(360, 290, 75, 23))
        self.pushButton_plot.setObjectName(_fromUtf8("pushButton_plot"))

        self.retranslateUi(Temporal)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("accepted()")), Temporal.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("rejected()")), Temporal.reject)
        QtCore.QMetaObject.connectSlotsByName(Temporal)

    def retranslateUi(self, Temporal):
        Temporal.setWindowTitle(_translate("Temporal", "Dialog", None))
        self.pushButton_folder.setText(_translate("Temporal", "...", None))
        self.label_title.setText(_translate("Temporal", "<html><head/><body><p><span style=\" font-size:16pt;\">Temporal Analysis of Medium Resolution data</span></p></body></html>", None))
        self.label_multiband.setText(_translate("Temporal", "Input Landsat Images Folder", None))
        self.label_multiband_2.setText(_translate("Temporal", "Input Mask", None))
        self.comboBox_mask.setItemText(0, _translate("Temporal", "[Choose from a file..]", "Click to select File"))
        self.label_multiband_3.setText(_translate("Temporal", "Number of Classes", None))
        self.pushButton_plot.setText(_translate("Temporal", "Graphs", None))

