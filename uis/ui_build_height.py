# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'uis/ui_build_height.ui'
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

class Ui_BuildHeight(object):
    def setupUi(self, BuildHeight):
        BuildHeight.setObjectName(_fromUtf8("BuildHeight"))
        BuildHeight.resize(466, 432)
        BuildHeight.setMinimumSize(QtCore.QSize(466, 432))
        BuildHeight.setMaximumSize(QtCore.QSize(466, 16777215))
        self.buttonBox = QtGui.QDialogButtonBox(BuildHeight)
        self.buttonBox.setGeometry(QtCore.QRect(240, 370, 161, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.pushButton_input_buldings = QtGui.QPushButton(BuildHeight)
        self.pushButton_input_buldings.setGeometry(QtCore.QRect(340, 150, 98, 27))
        self.pushButton_input_buldings.setObjectName(_fromUtf8("pushButton_input_buldings"))
        self.label = QtGui.QLabel(BuildHeight)
        self.label.setGeometry(QtCore.QRect(30, 130, 141, 17))
        self.label.setObjectName(_fromUtf8("label"))
        self.pushButton_input_shadows = QtGui.QPushButton(BuildHeight)
        self.pushButton_input_shadows.setGeometry(QtCore.QRect(340, 203, 98, 27))
        self.pushButton_input_shadows.setObjectName(_fromUtf8("pushButton_input_shadows"))
        self.lineEdit_input_shadows = QtGui.QLineEdit(BuildHeight)
        self.lineEdit_input_shadows.setGeometry(QtCore.QRect(30, 203, 291, 27))
        self.lineEdit_input_shadows.setObjectName(_fromUtf8("lineEdit_input_shadows"))
        self.lineEdit_input_buildings = QtGui.QLineEdit(BuildHeight)
        self.lineEdit_input_buildings.setGeometry(QtCore.QRect(30, 150, 291, 27))
        self.lineEdit_input_buildings.setObjectName(_fromUtf8("lineEdit_input_buildings"))
        self.label_3 = QtGui.QLabel(BuildHeight)
        self.label_3.setGeometry(QtCore.QRect(30, 180, 151, 17))
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.dateTimeEdit = QtGui.QDateTimeEdit(BuildHeight)
        self.dateTimeEdit.setGeometry(QtCore.QRect(30, 330, 194, 22))
        self.dateTimeEdit.setCalendarPopup(True)
        self.dateTimeEdit.setObjectName(_fromUtf8("dateTimeEdit"))
        self.lineEdit_output = QtGui.QLineEdit(BuildHeight)
        self.lineEdit_output.setGeometry(QtCore.QRect(30, 260, 291, 21))
        self.lineEdit_output.setObjectName(_fromUtf8("lineEdit_output"))
        self.label_1 = QtGui.QLabel(BuildHeight)
        self.label_1.setGeometry(QtCore.QRect(30, 240, 151, 17))
        self.label_1.setObjectName(_fromUtf8("label_1"))
        self.pushButton_output = QtGui.QPushButton(BuildHeight)
        self.pushButton_output.setGeometry(QtCore.QRect(340, 260, 98, 27))
        self.pushButton_output.setObjectName(_fromUtf8("pushButton_output"))
        self.lineEdit_shadow_field = QtGui.QLineEdit(BuildHeight)
        self.lineEdit_shadow_field.setGeometry(QtCore.QRect(240, 330, 113, 21))
        self.lineEdit_shadow_field.setObjectName(_fromUtf8("lineEdit_shadow_field"))
        self.label_2 = QtGui.QLabel(BuildHeight)
        self.label_2.setGeometry(QtCore.QRect(240, 310, 101, 17))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.doubleSpinBox_window_paramater = QtGui.QDoubleSpinBox(BuildHeight)
        self.doubleSpinBox_window_paramater.setGeometry(QtCore.QRect(30, 380, 71, 21))
        self.doubleSpinBox_window_paramater.setPrefix(_fromUtf8(""))
        self.doubleSpinBox_window_paramater.setSuffix(_fromUtf8(""))
        self.doubleSpinBox_window_paramater.setDecimals(2)
        self.doubleSpinBox_window_paramater.setMaximum(1000000.0)
        self.doubleSpinBox_window_paramater.setSingleStep(0.5)
        self.doubleSpinBox_window_paramater.setProperty("value", 1.0)
        self.doubleSpinBox_window_paramater.setObjectName(_fromUtf8("doubleSpinBox_window_paramater"))
        self.label_4 = QtGui.QLabel(BuildHeight)
        self.label_4.setGeometry(QtCore.QRect(30, 310, 201, 16))
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.label_5 = QtGui.QLabel(BuildHeight)
        self.label_5.setGeometry(QtCore.QRect(30, 360, 181, 17))
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.label_title = QtGui.QLabel(BuildHeight)
        self.label_title.setGeometry(QtCore.QRect(190, 100, 81, 21))
        self.label_title.setObjectName(_fromUtf8("label_title"))
        self.logo_eucentre = QtGui.QLabel(BuildHeight)
        self.logo_eucentre.setGeometry(QtCore.QRect(270, 10, 71, 81))
        self.logo_eucentre.setText(_fromUtf8(""))
        self.logo_eucentre.setPixmap(QtGui.QPixmap(_fromUtf8("eucentre.png")))
        self.logo_eucentre.setObjectName(_fromUtf8("logo_eucentre"))
        self.logo_sensum = QtGui.QLabel(BuildHeight)
        self.logo_sensum.setGeometry(QtCore.QRect(10, 20, 241, 61))
        self.logo_sensum.setText(_fromUtf8(""))
        self.logo_sensum.setPixmap(QtGui.QPixmap(_fromUtf8("sensum.png")))
        self.logo_sensum.setObjectName(_fromUtf8("logo_sensum"))
        self.logo_unipv = QtGui.QLabel(BuildHeight)
        self.logo_unipv.setGeometry(QtCore.QRect(350, 10, 91, 81))
        self.logo_unipv.setText(_fromUtf8(""))
        self.logo_unipv.setPixmap(QtGui.QPixmap(_fromUtf8("unipv.png")))
        self.logo_unipv.setObjectName(_fromUtf8("logo_unipv"))

        self.retranslateUi(BuildHeight)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("accepted()")), BuildHeight.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("rejected()")), BuildHeight.reject)
        QtCore.QMetaObject.connectSlotsByName(BuildHeight)

    def retranslateUi(self, BuildHeight):
        BuildHeight.setWindowTitle(_translate("BuildHeight", "SensumTools", None))
        self.pushButton_input_buldings.setText(_translate("BuildHeight", "...", None))
        self.label.setText(_translate("BuildHeight", "Building ESRI Shapefile", None))
        self.pushButton_input_shadows.setText(_translate("BuildHeight", "...", None))
        self.label_3.setText(_translate("BuildHeight", "Shadow ESRI Shapefile", None))
        self.dateTimeEdit.setDisplayFormat(_translate("BuildHeight", "yyyy/M/d hh:mm:ss", None))
        self.label_1.setText(_translate("BuildHeight", "Output ESRI Shapefile", None))
        self.pushButton_output.setText(_translate("BuildHeight", "...", None))
        self.lineEdit_shadow_field.setText(_translate("BuildHeight", "ID", None))
        self.label_2.setText(_translate("BuildHeight", "Shadow ID Field", None))
        self.label_4.setText(_translate("BuildHeight", "<html><head/><body>Acquisition date <span style=\" font-size:6pt;\">(hour in EDT)</span></body></html>", None))
        self.label_5.setText(_translate("BuildHeight", "Window Paramater", None))
        self.label_title.setText(_translate("BuildHeight", "<html><head/><body><p><span style=\" font-size:16pt;\">HEIGHT</span></p></body></html>", None))

