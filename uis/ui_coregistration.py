# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'uis/ui_coregistration.ui'
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

class Ui_Coregistration(object):
    def setupUi(self, Coregistration):
        Coregistration.setObjectName(_fromUtf8("Coregistration"))
        Coregistration.resize(464, 286)
        Coregistration.setMinimumSize(QtCore.QSize(464, 286))
        Coregistration.setMaximumSize(QtCore.QSize(464, 286))
        self.buttonBox = QtGui.QDialogButtonBox(Coregistration)
        self.buttonBox.setGeometry(QtCore.QRect(10, 240, 171, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.lineEdit_reference = QtGui.QLineEdit(Coregistration)
        self.lineEdit_reference.setGeometry(QtCore.QRect(20, 200, 291, 27))
        self.lineEdit_reference.setObjectName(_fromUtf8("lineEdit_reference"))
        self.label_reference = QtGui.QLabel(Coregistration)
        self.label_reference.setGeometry(QtCore.QRect(20, 180, 131, 17))
        self.label_reference.setObjectName(_fromUtf8("label_reference"))
        self.pushButton_reference = QtGui.QPushButton(Coregistration)
        self.pushButton_reference.setGeometry(QtCore.QRect(330, 200, 98, 27))
        self.pushButton_reference.setObjectName(_fromUtf8("pushButton_reference"))
        self.label_tobechange = QtGui.QLabel(Coregistration)
        self.label_tobechange.setGeometry(QtCore.QRect(20, 132, 191, 17))
        self.label_tobechange.setObjectName(_fromUtf8("label_tobechange"))
        self.pushButton_tobechange = QtGui.QPushButton(Coregistration)
        self.pushButton_tobechange.setGeometry(QtCore.QRect(330, 150, 98, 27))
        self.pushButton_tobechange.setObjectName(_fromUtf8("pushButton_tobechange"))
        self.lineEdit_tobechange = QtGui.QLineEdit(Coregistration)
        self.lineEdit_tobechange.setGeometry(QtCore.QRect(20, 150, 291, 27))
        self.lineEdit_tobechange.setObjectName(_fromUtf8("lineEdit_tobechange"))
        self.label_title = QtGui.QLabel(Coregistration)
        self.label_title.setGeometry(QtCore.QRect(120, 100, 201, 21))
        self.label_title.setObjectName(_fromUtf8("label_title"))
        self.logo_eucentre = QtGui.QLabel(Coregistration)
        self.logo_eucentre.setGeometry(QtCore.QRect(270, 10, 71, 81))
        self.logo_eucentre.setText(_fromUtf8(""))
        self.logo_eucentre.setPixmap(QtGui.QPixmap(_fromUtf8("eucentre.png")))
        self.logo_eucentre.setObjectName(_fromUtf8("logo_eucentre"))
        self.logo_sensum = QtGui.QLabel(Coregistration)
        self.logo_sensum.setGeometry(QtCore.QRect(10, 20, 241, 61))
        self.logo_sensum.setText(_fromUtf8(""))
        self.logo_sensum.setPixmap(QtGui.QPixmap(_fromUtf8("sensum.png")))
        self.logo_sensum.setObjectName(_fromUtf8("logo_sensum"))
        self.logo_unipv = QtGui.QLabel(Coregistration)
        self.logo_unipv.setGeometry(QtCore.QRect(350, 10, 91, 81))
        self.logo_unipv.setText(_fromUtf8(""))
        self.logo_unipv.setPixmap(QtGui.QPixmap(_fromUtf8("unipv.png")))
        self.logo_unipv.setObjectName(_fromUtf8("logo_unipv"))

        self.retranslateUi(Coregistration)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("accepted()")), Coregistration.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("rejected()")), Coregistration.reject)
        QtCore.QMetaObject.connectSlotsByName(Coregistration)

    def retranslateUi(self, Coregistration):
        Coregistration.setWindowTitle(_translate("Coregistration", "SensumTools", None))
        self.label_reference.setText(_translate("Coregistration", "References directory", None))
        self.pushButton_reference.setText(_translate("Coregistration", "...", None))
        self.label_tobechange.setText(_translate("Coregistration", "Main Directory", None))
        self.pushButton_tobechange.setText(_translate("Coregistration", "...", None))
        self.label_title.setText(_translate("Coregistration", "<html><head/><body><p><span style=\" font-size:16pt;\">CO-REGISTRATION</span></p></body></html>", None))

