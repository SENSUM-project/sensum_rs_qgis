# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'uis/ui_footprints.ui'
#
# Created: Wed Jun 11 12:28:26 2014
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

class Ui_Footprints(object):
    def setupUi(self, Footprints):
        Footprints.setObjectName(_fromUtf8("Footprints"))
        Footprints.resize(391, 428)
        Footprints.setMinimumSize(QtCore.QSize(391, 428))
        Footprints.setMaximumSize(QtCore.QSize(391, 428))
        self.buttonBox = QtGui.QDialogButtonBox(Footprints)
        self.buttonBox.setGeometry(QtCore.QRect(220, 390, 161, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.listWidget = QtGui.QListWidget(Footprints)
        self.listWidget.setGeometry(QtCore.QRect(50, 250, 181, 81))
        self.listWidget.setObjectName(_fromUtf8("listWidget"))
        self.label_pansharp = QtGui.QLabel(Footprints)
        self.label_pansharp.setGeometry(QtCore.QRect(30, 50, 58, 15))
        self.label_pansharp.setObjectName(_fromUtf8("label_pansharp"))
        self.lineEdit_classes = QtGui.QLineEdit(Footprints)
        self.lineEdit_classes.setGeometry(QtCore.QRect(50, 340, 181, 21))
        self.lineEdit_classes.setObjectName(_fromUtf8("lineEdit_classes"))
        self.lineEdit_pansharp = QtGui.QLineEdit(Footprints)
        self.lineEdit_pansharp.setGeometry(QtCore.QRect(30, 70, 281, 21))
        self.lineEdit_pansharp.setObjectName(_fromUtf8("lineEdit_pansharp"))
        self.label_training = QtGui.QLabel(Footprints)
        self.label_training.setGeometry(QtCore.QRect(30, 110, 51, 16))
        self.label_training.setObjectName(_fromUtf8("label_training"))
        self.lineEdit_training = QtGui.QLineEdit(Footprints)
        self.lineEdit_training.setGeometry(QtCore.QRect(30, 130, 281, 21))
        self.lineEdit_training.setObjectName(_fromUtf8("lineEdit_training"))
        self.label_training_field = QtGui.QLabel(Footprints)
        self.label_training_field.setGeometry(QtCore.QRect(30, 170, 81, 16))
        self.label_training_field.setObjectName(_fromUtf8("label_training_field"))
        self.lineEdit_training_field = QtGui.QLineEdit(Footprints)
        self.lineEdit_training_field.setGeometry(QtCore.QRect(30, 190, 131, 21))
        self.lineEdit_training_field.setObjectName(_fromUtf8("lineEdit_training_field"))
        self.label_training_2 = QtGui.QLabel(Footprints)
        self.label_training_2.setGeometry(QtCore.QRect(90, 230, 101, 16))
        self.label_training_2.setObjectName(_fromUtf8("label_training_2"))
        self.pushButton_clear = QtGui.QPushButton(Footprints)
        self.pushButton_clear.setGeometry(QtCore.QRect(240, 250, 101, 25))
        self.pushButton_clear.setObjectName(_fromUtf8("pushButton_clear"))
        self.pushButton_pansharp = QtGui.QPushButton(Footprints)
        self.pushButton_pansharp.setGeometry(QtCore.QRect(320, 70, 41, 21))
        self.pushButton_pansharp.setObjectName(_fromUtf8("pushButton_pansharp"))
        self.pushButton_training = QtGui.QPushButton(Footprints)
        self.pushButton_training.setGeometry(QtCore.QRect(320, 130, 41, 21))
        self.pushButton_training.setObjectName(_fromUtf8("pushButton_training"))
        self.pushButton_add = QtGui.QPushButton(Footprints)
        self.pushButton_add.setGeometry(QtCore.QRect(240, 340, 41, 25))
        self.pushButton_add.setObjectName(_fromUtf8("pushButton_add"))
        self.pushButton_clear_all = QtGui.QPushButton(Footprints)
        self.pushButton_clear_all.setGeometry(QtCore.QRect(240, 280, 101, 25))
        self.pushButton_clear_all.setObjectName(_fromUtf8("pushButton_clear_all"))
        self.label_title = QtGui.QLabel(Footprints)
        self.label_title.setGeometry(QtCore.QRect(70, 20, 251, 21))
        self.label_title.setObjectName(_fromUtf8("label_title"))

        self.retranslateUi(Footprints)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("accepted()")), Footprints.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("rejected()")), Footprints.reject)
        QtCore.QMetaObject.connectSlotsByName(Footprints)

    def retranslateUi(self, Footprints):
        Footprints.setWindowTitle(_translate("Footprints", "Dialog", None))
        self.label_pansharp.setText(_translate("Footprints", "Pansharp", None))
        self.label_training.setText(_translate("Footprints", "Training", None))
        self.label_training_field.setText(_translate("Footprints", "Training Field", None))
        self.lineEdit_training_field.setText(_translate("Footprints", "Class", None))
        self.label_training_2.setText(_translate("Footprints", "Building Classes", None))
        self.pushButton_clear.setText(_translate("Footprints", "Clear Selected", None))
        self.pushButton_pansharp.setText(_translate("Footprints", "...", None))
        self.pushButton_training.setText(_translate("Footprints", "...", None))
        self.pushButton_add.setText(_translate("Footprints", "Add", None))
        self.pushButton_clear_all.setText(_translate("Footprints", "Clear All", None))
        self.label_title.setText(_translate("Footprints", "<html><head/><body><p><span style=\" font-size:16pt;\">BUILDING EXTRACTION</span></p></body></html>", None))

