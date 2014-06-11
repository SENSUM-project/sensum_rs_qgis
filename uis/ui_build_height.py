# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'uis/ui_build_height.ui'
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

class Ui_BuildHeight(object):
    def setupUi(self, BuildHeight):
        BuildHeight.setObjectName(_fromUtf8("BuildHeight"))
        BuildHeight.resize(466, 351)
        BuildHeight.setMinimumSize(QtCore.QSize(466, 351))
        BuildHeight.setMaximumSize(QtCore.QSize(466, 351))
        self.buttonBox = QtGui.QDialogButtonBox(BuildHeight)
        self.buttonBox.setGeometry(QtCore.QRect(240, 290, 161, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.pushButton_input_buldings = QtGui.QPushButton(BuildHeight)
        self.pushButton_input_buldings.setGeometry(QtCore.QRect(340, 70, 98, 27))
        self.pushButton_input_buldings.setObjectName(_fromUtf8("pushButton_input_buldings"))
        self.label = QtGui.QLabel(BuildHeight)
        self.label.setGeometry(QtCore.QRect(30, 50, 141, 17))
        self.label.setObjectName(_fromUtf8("label"))
        self.pushButton_input_shadows = QtGui.QPushButton(BuildHeight)
        self.pushButton_input_shadows.setGeometry(QtCore.QRect(340, 123, 98, 27))
        self.pushButton_input_shadows.setObjectName(_fromUtf8("pushButton_input_shadows"))
        self.lineEdit_input_shadows = QtGui.QLineEdit(BuildHeight)
        self.lineEdit_input_shadows.setGeometry(QtCore.QRect(30, 123, 291, 27))
        self.lineEdit_input_shadows.setObjectName(_fromUtf8("lineEdit_input_shadows"))
        self.lineEdit_input_buildings = QtGui.QLineEdit(BuildHeight)
        self.lineEdit_input_buildings.setGeometry(QtCore.QRect(30, 70, 291, 27))
        self.lineEdit_input_buildings.setObjectName(_fromUtf8("lineEdit_input_buildings"))
        self.label_3 = QtGui.QLabel(BuildHeight)
        self.label_3.setGeometry(QtCore.QRect(30, 100, 151, 17))
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.dateTimeEdit = QtGui.QDateTimeEdit(BuildHeight)
        self.dateTimeEdit.setGeometry(QtCore.QRect(30, 250, 194, 22))
        self.dateTimeEdit.setCalendarPopup(True)
        self.dateTimeEdit.setObjectName(_fromUtf8("dateTimeEdit"))
        self.lineEdit_output = QtGui.QLineEdit(BuildHeight)
        self.lineEdit_output.setGeometry(QtCore.QRect(30, 180, 291, 21))
        self.lineEdit_output.setObjectName(_fromUtf8("lineEdit_output"))
        self.label_1 = QtGui.QLabel(BuildHeight)
        self.label_1.setGeometry(QtCore.QRect(30, 160, 151, 17))
        self.label_1.setObjectName(_fromUtf8("label_1"))
        self.pushButton_output = QtGui.QPushButton(BuildHeight)
        self.pushButton_output.setGeometry(QtCore.QRect(340, 180, 98, 27))
        self.pushButton_output.setObjectName(_fromUtf8("pushButton_output"))
        self.lineEdit_shadow_field = QtGui.QLineEdit(BuildHeight)
        self.lineEdit_shadow_field.setGeometry(QtCore.QRect(240, 250, 113, 21))
        self.lineEdit_shadow_field.setObjectName(_fromUtf8("lineEdit_shadow_field"))
        self.label_2 = QtGui.QLabel(BuildHeight)
        self.label_2.setGeometry(QtCore.QRect(240, 230, 101, 17))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.doubleSpinBox_window_paramater = QtGui.QDoubleSpinBox(BuildHeight)
        self.doubleSpinBox_window_paramater.setGeometry(QtCore.QRect(30, 300, 71, 21))
        self.doubleSpinBox_window_paramater.setPrefix(_fromUtf8(""))
        self.doubleSpinBox_window_paramater.setSuffix(_fromUtf8(""))
        self.doubleSpinBox_window_paramater.setDecimals(2)
        self.doubleSpinBox_window_paramater.setMaximum(1000000.0)
        self.doubleSpinBox_window_paramater.setSingleStep(0.5)
        self.doubleSpinBox_window_paramater.setProperty("value", 1.0)
        self.doubleSpinBox_window_paramater.setObjectName(_fromUtf8("doubleSpinBox_window_paramater"))
        self.label_4 = QtGui.QLabel(BuildHeight)
        self.label_4.setGeometry(QtCore.QRect(30, 230, 201, 16))
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.label_5 = QtGui.QLabel(BuildHeight)
        self.label_5.setGeometry(QtCore.QRect(30, 280, 181, 17))
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.label_title = QtGui.QLabel(BuildHeight)
        self.label_title.setGeometry(QtCore.QRect(190, 20, 81, 21))
        self.label_title.setObjectName(_fromUtf8("label_title"))

        self.retranslateUi(BuildHeight)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("accepted()")), BuildHeight.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("rejected()")), BuildHeight.reject)
        QtCore.QMetaObject.connectSlotsByName(BuildHeight)

    def retranslateUi(self, BuildHeight):
        BuildHeight.setWindowTitle(_translate("BuildHeight", "Dialog", None))
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

