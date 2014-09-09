# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'uis/ui_change_detection.ui'
#
# Created: Wed Sep  3 17:20:58 2014
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

class Ui_ChangeDetection(object):
    def setupUi(self, ChangeDetection):
        ChangeDetection.setObjectName(_fromUtf8("ChangeDetection"))
        ChangeDetection.resize(496, 330)
        ChangeDetection.setMinimumSize(QtCore.QSize(496, 330))
        ChangeDetection.setMaximumSize(QtCore.QSize(496, 330))
        self.buttonBox = QtGui.QDialogButtonBox(ChangeDetection)
        self.buttonBox.setGeometry(QtCore.QRect(260, 260, 171, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.label_tobechange = QtGui.QLabel(ChangeDetection)
        self.label_tobechange.setGeometry(QtCore.QRect(40, 132, 191, 17))
        self.label_tobechange.setObjectName(_fromUtf8("label_tobechange"))
        self.pushButton_tobechange = QtGui.QPushButton(ChangeDetection)
        self.pushButton_tobechange.setGeometry(QtCore.QRect(350, 150, 98, 27))
        self.pushButton_tobechange.setObjectName(_fromUtf8("pushButton_tobechange"))
        self.lineEdit_tobechange = QtGui.QLineEdit(ChangeDetection)
        self.lineEdit_tobechange.setGeometry(QtCore.QRect(40, 150, 291, 27))
        self.lineEdit_tobechange.setObjectName(_fromUtf8("lineEdit_tobechange"))
        self.label_title = QtGui.QLabel(ChangeDetection)
        self.label_title.setGeometry(QtCore.QRect(150, 100, 201, 21))
        self.label_title.setObjectName(_fromUtf8("label_title"))
        self.logo_eucentre = QtGui.QLabel(ChangeDetection)
        self.logo_eucentre.setGeometry(QtCore.QRect(270, 10, 71, 81))
        self.logo_eucentre.setText(_fromUtf8(""))
        self.logo_eucentre.setPixmap(QtGui.QPixmap(_fromUtf8(".sensum/eucentre.png")))
        self.logo_eucentre.setObjectName(_fromUtf8("logo_eucentre"))
        self.logo_sensum = QtGui.QLabel(ChangeDetection)
        self.logo_sensum.setGeometry(QtCore.QRect(10, 20, 241, 61))
        self.logo_sensum.setText(_fromUtf8(""))
        self.logo_sensum.setPixmap(QtGui.QPixmap(_fromUtf8(".sensum/sensum.png")))
        self.logo_sensum.setObjectName(_fromUtf8("logo_sensum"))
        self.logo_unipv = QtGui.QLabel(ChangeDetection)
        self.logo_unipv.setGeometry(QtCore.QRect(350, 10, 91, 81))
        self.logo_unipv.setText(_fromUtf8(""))
        self.logo_unipv.setPixmap(QtGui.QPixmap(_fromUtf8(".sensum/unipv.png")))
        self.logo_unipv.setObjectName(_fromUtf8("logo_unipv"))
        self.label_tobechange_2 = QtGui.QLabel(ChangeDetection)
        self.label_tobechange_2.setGeometry(QtCore.QRect(60, 200, 61, 20))
        self.label_tobechange_2.setObjectName(_fromUtf8("label_tobechange_2"))
        self.lineEdit_field = QtGui.QLineEdit(ChangeDetection)
        self.lineEdit_field.setGeometry(QtCore.QRect(60, 220, 161, 21))
        self.lineEdit_field.setObjectName(_fromUtf8("lineEdit_field"))
        self.comboBox_extraction = QtGui.QComboBox(ChangeDetection)
        self.comboBox_extraction.setGeometry(QtCore.QRect(270, 210, 111, 22))
        self.comboBox_extraction.setObjectName(_fromUtf8("comboBox_extraction"))
        self.comboBox_extraction.addItem(_fromUtf8(""))
        self.comboBox_extraction.addItem(_fromUtf8(""))

        self.retranslateUi(ChangeDetection)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("accepted()")), ChangeDetection.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("rejected()")), ChangeDetection.reject)
        QtCore.QMetaObject.connectSlotsByName(ChangeDetection)

    def retranslateUi(self, ChangeDetection):
        ChangeDetection.setWindowTitle(_translate("ChangeDetection", "SensumTools", None))
        self.label_tobechange.setText(_translate("ChangeDetection", "Main Directory", None))
        self.pushButton_tobechange.setText(_translate("ChangeDetection", "...", None))
        self.label_title.setText(_translate("ChangeDetection", "<html><head/><body><p><span style=\" font-size:16pt;\">CHANGE DETECTION</span></p></body></html>", None))
        self.label_tobechange_2.setText(_translate("ChangeDetection", "Urban Field", None))
        self.lineEdit_field.setText(_translate("ChangeDetection", "UrbanClass", None))
        self.comboBox_extraction.setItemText(0, _translate("ChangeDetection", "Dissimilarity-Based", None))
        self.comboBox_extraction.setItemText(1, _translate("ChangeDetection", "PCA-Based", None))

