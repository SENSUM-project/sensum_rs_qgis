# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_coregistration.ui'
#
# Created: Wed Nov 26 12:44:35 2014
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

class Ui_Coregistration(object):
    def setupUi(self, Coregistration):
        Coregistration.setObjectName(_fromUtf8("Coregistration"))
        Coregistration.resize(500, 463)
        Coregistration.setMinimumSize(QtCore.QSize(500, 463))
        Coregistration.setMaximumSize(QtCore.QSize(500, 463))
        self.buttonBox = QtGui.QDialogButtonBox(Coregistration)
        self.buttonBox.setGeometry(QtCore.QRect(260, 380, 171, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.lineEdit_reference = QtGui.QLineEdit(Coregistration)
        self.lineEdit_reference.setGeometry(QtCore.QRect(40, 200, 291, 27))
        self.lineEdit_reference.setObjectName(_fromUtf8("lineEdit_reference"))
        self.label_reference = QtGui.QLabel(Coregistration)
        self.label_reference.setGeometry(QtCore.QRect(40, 180, 131, 17))
        self.label_reference.setObjectName(_fromUtf8("label_reference"))
        self.pushButton_reference = QtGui.QPushButton(Coregistration)
        self.pushButton_reference.setGeometry(QtCore.QRect(350, 200, 98, 27))
        self.pushButton_reference.setObjectName(_fromUtf8("pushButton_reference"))
        self.label_tobechange = QtGui.QLabel(Coregistration)
        self.label_tobechange.setGeometry(QtCore.QRect(40, 132, 191, 17))
        self.label_tobechange.setObjectName(_fromUtf8("label_tobechange"))
        self.pushButton_tobechange = QtGui.QPushButton(Coregistration)
        self.pushButton_tobechange.setGeometry(QtCore.QRect(350, 150, 98, 27))
        self.pushButton_tobechange.setObjectName(_fromUtf8("pushButton_tobechange"))
        self.lineEdit_tobechange = QtGui.QLineEdit(Coregistration)
        self.lineEdit_tobechange.setGeometry(QtCore.QRect(40, 150, 291, 27))
        self.lineEdit_tobechange.setObjectName(_fromUtf8("lineEdit_tobechange"))
        self.label_title = QtGui.QLabel(Coregistration)
        self.label_title.setGeometry(QtCore.QRect(150, 100, 201, 21))
        self.label_title.setObjectName(_fromUtf8("label_title"))
        self.logo_eucentre = QtGui.QLabel(Coregistration)
        self.logo_eucentre.setGeometry(QtCore.QRect(270, 10, 71, 81))
        self.logo_eucentre.setText(_fromUtf8(""))
        self.logo_eucentre.setPixmap(QtGui.QPixmap(_fromUtf8(".sensum/eucentre.png")))
        self.logo_eucentre.setObjectName(_fromUtf8("logo_eucentre"))
        self.logo_sensum = QtGui.QLabel(Coregistration)
        self.logo_sensum.setGeometry(QtCore.QRect(10, 20, 241, 61))
        self.logo_sensum.setText(_fromUtf8(""))
        self.logo_sensum.setPixmap(QtGui.QPixmap(_fromUtf8(".sensum/sensum.png")))
        self.logo_sensum.setObjectName(_fromUtf8("logo_sensum"))
        self.logo_unipv = QtGui.QLabel(Coregistration)
        self.logo_unipv.setGeometry(QtCore.QRect(350, 10, 91, 81))
        self.logo_unipv.setText(_fromUtf8(""))
        self.logo_unipv.setPixmap(QtGui.QPixmap(_fromUtf8(".sensum/unipv.png")))
        self.logo_unipv.setObjectName(_fromUtf8("logo_unipv"))
        self.comboBox_select_crop = QtGui.QComboBox(Coregistration)
        self.comboBox_select_crop.setGeometry(QtCore.QRect(40, 240, 161, 22))
        self.comboBox_select_crop.setObjectName(_fromUtf8("comboBox_select_crop"))
        self.comboBox_select_crop.addItem(_fromUtf8(""))
        self.comboBox_select_crop.addItem(_fromUtf8(""))
        self.comboBox_select_crop.addItem(_fromUtf8(""))
        self.groupBox_clip = QtGui.QGroupBox(Coregistration)
        self.groupBox_clip.setGeometry(QtCore.QRect(20, 280, 451, 91))
        self.groupBox_clip.setObjectName(_fromUtf8("groupBox_clip"))
        self.label_tobechange_2 = QtGui.QLabel(self.groupBox_clip)
        self.label_tobechange_2.setGeometry(QtCore.QRect(30, 30, 81, 17))
        self.label_tobechange_2.setObjectName(_fromUtf8("label_tobechange_2"))
        self.comboBox_input_shape = QtGui.QComboBox(self.groupBox_clip)
        self.comboBox_input_shape.setGeometry(QtCore.QRect(30, 50, 371, 20))
        self.comboBox_input_shape.setObjectName(_fromUtf8("comboBox_input_shape"))
        self.comboBox_input_shape.addItem(_fromUtf8(""))
        self.groupBox_grid = QtGui.QGroupBox(Coregistration)
        self.groupBox_grid.setGeometry(QtCore.QRect(20, 280, 451, 91))
        self.groupBox_grid.setObjectName(_fromUtf8("groupBox_grid"))
        self.spinBox_rows = QtGui.QSpinBox(self.groupBox_grid)
        self.spinBox_rows.setGeometry(QtCore.QRect(150, 40, 42, 22))
        self.spinBox_rows.setProperty("value", 6)
        self.spinBox_rows.setObjectName(_fromUtf8("spinBox_rows"))
        self.spinBox_cols = QtGui.QSpinBox(self.groupBox_grid)
        self.spinBox_cols.setGeometry(QtCore.QRect(290, 40, 42, 22))
        self.spinBox_cols.setProperty("value", 6)
        self.spinBox_cols.setObjectName(_fromUtf8("spinBox_cols"))
        self.label_tobechange_3 = QtGui.QLabel(self.groupBox_grid)
        self.label_tobechange_3.setGeometry(QtCore.QRect(110, 40, 31, 17))
        self.label_tobechange_3.setObjectName(_fromUtf8("label_tobechange_3"))
        self.label_tobechange_4 = QtGui.QLabel(self.groupBox_grid)
        self.label_tobechange_4.setGeometry(QtCore.QRect(250, 40, 31, 17))
        self.label_tobechange_4.setObjectName(_fromUtf8("label_tobechange_4"))
        self.checkBox_surf = QtGui.QCheckBox(Coregistration)
        self.checkBox_surf.setGeometry(QtCore.QRect(260, 250, 70, 17))
        self.checkBox_surf.setObjectName(_fromUtf8("checkBox_surf"))
        self.checkBox_fft = QtGui.QCheckBox(Coregistration)
        self.checkBox_fft.setGeometry(QtCore.QRect(330, 250, 70, 17))
        self.checkBox_fft.setObjectName(_fromUtf8("checkBox_fft"))
        self.groupBox_unsupervised_classification = QtGui.QGroupBox(Coregistration)
        self.groupBox_unsupervised_classification.setGeometry(QtCore.QRect(20, 280, 451, 91))
        self.groupBox_unsupervised_classification.setObjectName(_fromUtf8("groupBox_unsupervised_classification"))
        self.spinBox_nclasses = QtGui.QSpinBox(self.groupBox_unsupervised_classification)
        self.spinBox_nclasses.setGeometry(QtCore.QRect(230, 40, 42, 22))
        self.spinBox_nclasses.setProperty("value", 5)
        self.spinBox_nclasses.setObjectName(_fromUtf8("spinBox_nclasses"))
        self.label_tobechange_6 = QtGui.QLabel(self.groupBox_unsupervised_classification)
        self.label_tobechange_6.setGeometry(QtCore.QRect(170, 40, 51, 20))
        self.label_tobechange_6.setObjectName(_fromUtf8("label_tobechange_6"))

        self.retranslateUi(Coregistration)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("accepted()")), Coregistration.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("rejected()")), Coregistration.reject)
        QtCore.QMetaObject.connectSlotsByName(Coregistration)

    def retranslateUi(self, Coregistration):
        Coregistration.setWindowTitle(_translate("Coregistration", "SensumTools", None))
        self.label_reference.setText(_translate("Coregistration", "Reference directory", None))
        self.pushButton_reference.setText(_translate("Coregistration", "...", None))
        self.label_tobechange.setText(_translate("Coregistration", "Main Directory", None))
        self.pushButton_tobechange.setText(_translate("Coregistration", "...", None))
        self.label_title.setText(_translate("Coregistration", "<html><head/><body><p><span style=\" font-size:16pt;\">CO-REGISTRATION</span></p></body></html>", None))
        self.comboBox_select_crop.setItemText(0, _translate("Coregistration", "Clip", None))
        self.comboBox_select_crop.setItemText(1, _translate("Coregistration", "Grid", None))
        self.comboBox_select_crop.setItemText(2, _translate("Coregistration", "Unsupervised Classification", None))
        self.groupBox_clip.setTitle(_translate("Coregistration", "Clip", None))
        self.label_tobechange_2.setText(_translate("Coregistration", "Input Shapefile", None))
        self.comboBox_input_shape.setItemText(0, _translate("Coregistration", "[Choose from a file..]", "Click to select File"))
        self.groupBox_grid.setTitle(_translate("Coregistration", "Grid", None))
        self.label_tobechange_3.setText(_translate("Coregistration", "Rows", None))
        self.label_tobechange_4.setText(_translate("Coregistration", "Cols", None))
        self.checkBox_surf.setText(_translate("Coregistration", "SURF", None))
        self.checkBox_fft.setText(_translate("Coregistration", "FFT", None))
        self.groupBox_unsupervised_classification.setTitle(_translate("Coregistration", "Unsupervised Classification", None))
        self.label_tobechange_6.setText(_translate("Coregistration", "N classes", None))

