# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_change_detection_dlks.ui'
#
# Created: Tue Oct 21 16:56:51 2014
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

class Ui_ChangeDetectionDilkushi(object):
    def setupUi(self, ChangeDetectionDilkushi):
        ChangeDetectionDilkushi.setObjectName(_fromUtf8("ChangeDetectionDilkushi"))
        ChangeDetectionDilkushi.resize(467, 493)
        self.widget = QtGui.QWidget(ChangeDetectionDilkushi)
        self.widget.setGeometry(QtCore.QRect(20, 140, 441, 301))
        self.widget.setObjectName(_fromUtf8("widget"))
        self.label_panchromatic = QtGui.QLabel(self.widget)
        self.label_panchromatic.setGeometry(QtCore.QRect(9, 63, 123, 16))
        self.label_panchromatic.setObjectName(_fromUtf8("label_panchromatic"))
        self.label_multiband = QtGui.QLabel(self.widget)
        self.label_multiband.setGeometry(QtCore.QRect(9, 9, 101, 16))
        self.label_multiband.setObjectName(_fromUtf8("label_multiband"))
        self.comboBox_multiband_pre = QtGui.QComboBox(self.widget)
        self.comboBox_multiband_pre.setGeometry(QtCore.QRect(10, 30, 411, 20))
        self.comboBox_multiband_pre.setObjectName(_fromUtf8("comboBox_multiband_pre"))
        self.comboBox_multiband_pre.addItem(_fromUtf8(""))
        self.comboBox_panchromatic_pre = QtGui.QComboBox(self.widget)
        self.comboBox_panchromatic_pre.setGeometry(QtCore.QRect(10, 90, 411, 20))
        self.comboBox_panchromatic_pre.setObjectName(_fromUtf8("comboBox_panchromatic_pre"))
        self.comboBox_panchromatic_pre.addItem(_fromUtf8(""))
        self.comboBox_multiband_post = QtGui.QComboBox(self.widget)
        self.comboBox_multiband_post.setGeometry(QtCore.QRect(11, 147, 411, 20))
        self.comboBox_multiband_post.setObjectName(_fromUtf8("comboBox_multiband_post"))
        self.comboBox_multiband_post.addItem(_fromUtf8(""))
        self.label_panchromatic_2 = QtGui.QLabel(self.widget)
        self.label_panchromatic_2.setGeometry(QtCore.QRect(10, 120, 123, 16))
        self.label_panchromatic_2.setObjectName(_fromUtf8("label_panchromatic_2"))
        self.comboBox_panchromatic_post = QtGui.QComboBox(self.widget)
        self.comboBox_panchromatic_post.setGeometry(QtCore.QRect(11, 210, 411, 20))
        self.comboBox_panchromatic_post.setObjectName(_fromUtf8("comboBox_panchromatic_post"))
        self.comboBox_panchromatic_post.addItem(_fromUtf8(""))
        self.label_panchromatic_3 = QtGui.QLabel(self.widget)
        self.label_panchromatic_3.setGeometry(QtCore.QRect(10, 183, 123, 16))
        self.label_panchromatic_3.setObjectName(_fromUtf8("label_panchromatic_3"))
        self.comboBox_clip_shapefile = QtGui.QComboBox(self.widget)
        self.comboBox_clip_shapefile.setGeometry(QtCore.QRect(11, 270, 411, 20))
        self.comboBox_clip_shapefile.setObjectName(_fromUtf8("comboBox_clip_shapefile"))
        self.comboBox_clip_shapefile.addItem(_fromUtf8(""))
        self.label_panchromatic_4 = QtGui.QLabel(self.widget)
        self.label_panchromatic_4.setGeometry(QtCore.QRect(10, 243, 123, 16))
        self.label_panchromatic_4.setObjectName(_fromUtf8("label_panchromatic_4"))
        self.label_title = QtGui.QLabel(ChangeDetectionDilkushi)
        self.label_title.setGeometry(QtCore.QRect(130, 100, 211, 21))
        self.label_title.setObjectName(_fromUtf8("label_title"))
        self.buttonBox = QtGui.QDialogButtonBox(ChangeDetectionDilkushi)
        self.buttonBox.setGeometry(QtCore.QRect(160, 450, 160, 27))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.logo_unipv = QtGui.QLabel(ChangeDetectionDilkushi)
        self.logo_unipv.setGeometry(QtCore.QRect(340, 10, 91, 81))
        self.logo_unipv.setText(_fromUtf8(""))
        self.logo_unipv.setPixmap(QtGui.QPixmap(_fromUtf8(".sensum/cambridge.png")))
        self.logo_unipv.setObjectName(_fromUtf8("logo_unipv"))
        self.logo_sensum = QtGui.QLabel(ChangeDetectionDilkushi)
        self.logo_sensum.setGeometry(QtCore.QRect(30, 20, 241, 61))
        self.logo_sensum.setText(_fromUtf8(""))
        self.logo_sensum.setPixmap(QtGui.QPixmap(_fromUtf8(".sensum/sensum.png")))
        self.logo_sensum.setObjectName(_fromUtf8("logo_sensum"))

        self.retranslateUi(ChangeDetectionDilkushi)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("accepted()")), ChangeDetectionDilkushi.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("rejected()")), ChangeDetectionDilkushi.reject)
        QtCore.QMetaObject.connectSlotsByName(ChangeDetectionDilkushi)

    def retranslateUi(self, ChangeDetectionDilkushi):
        ChangeDetectionDilkushi.setWindowTitle(_translate("ChangeDetectionDilkushi", "SensumTools", None))
        self.label_panchromatic.setText(_translate("ChangeDetectionDilkushi", "Panchromatic Pre", None))
        self.label_multiband.setText(_translate("ChangeDetectionDilkushi", "Multiband Pre", None))
        self.comboBox_multiband_pre.setItemText(0, _translate("ChangeDetectionDilkushi", "[Choose from a file..]", "Click to select File"))
        self.comboBox_panchromatic_pre.setItemText(0, _translate("ChangeDetectionDilkushi", "[Choose from a file..]", "Click to select File"))
        self.comboBox_multiband_post.setItemText(0, _translate("ChangeDetectionDilkushi", "[Choose from a file..]", "Click to select File"))
        self.label_panchromatic_2.setText(_translate("ChangeDetectionDilkushi", "Multiband Post", None))
        self.comboBox_panchromatic_post.setItemText(0, _translate("ChangeDetectionDilkushi", "[Choose from a file..]", "Click to select File"))
        self.label_panchromatic_3.setText(_translate("ChangeDetectionDilkushi", "Panchromatic Post", None))
        self.comboBox_clip_shapefile.setItemText(0, _translate("ChangeDetectionDilkushi", "[Choose from a file..]", "Click to select File"))
        self.label_panchromatic_4.setText(_translate("ChangeDetectionDilkushi", "Clip Shapefile", None))
        self.label_title.setText(_translate("ChangeDetectionDilkushi", "<html><head/><body><p><span style=\" font-size:16pt;\">CHANGE DETECTION</span></p></body></html>", None))

