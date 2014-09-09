# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_temporalgraph.ui'
#
# Created: Mon Sep 08 17:00:52 2014
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

class Ui_TemporalGraph(object):
    def setupUi(self, TemporalGraph):
        TemporalGraph.setObjectName(_fromUtf8("TemporalGraph"))
        TemporalGraph.resize(578, 332)
        self.buttonBox = QtGui.QDialogButtonBox(TemporalGraph)
        self.buttonBox.setGeometry(QtCore.QRect(310, 270, 161, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.logo_sensum = QtGui.QLabel(TemporalGraph)
        self.logo_sensum.setGeometry(QtCore.QRect(70, 20, 241, 61))
        self.logo_sensum.setText(_fromUtf8(""))
        self.logo_sensum.setPixmap(QtGui.QPixmap(_fromUtf8(".sensum/sensum.png")))
        self.logo_sensum.setObjectName(_fromUtf8("logo_sensum"))
        self.lineEdit_folder = QtGui.QLineEdit(TemporalGraph)
        self.lineEdit_folder.setGeometry(QtCore.QRect(70, 171, 321, 21))
        self.lineEdit_folder.setObjectName(_fromUtf8("lineEdit_folder"))
        self.pushButton_folder = QtGui.QPushButton(TemporalGraph)
        self.pushButton_folder.setGeometry(QtCore.QRect(400, 171, 77, 21))
        self.pushButton_folder.setObjectName(_fromUtf8("pushButton_folder"))
        self.label_title = QtGui.QLabel(TemporalGraph)
        self.label_title.setGeometry(QtCore.QRect(150, 101, 251, 31))
        self.label_title.setObjectName(_fromUtf8("label_title"))
        self.label_multiband = QtGui.QLabel(TemporalGraph)
        self.label_multiband.setGeometry(QtCore.QRect(70, 151, 201, 16))
        self.label_multiband.setObjectName(_fromUtf8("label_multiband"))
        self.logo_unipv = QtGui.QLabel(TemporalGraph)
        self.logo_unipv.setGeometry(QtCore.QRect(380, 10, 91, 81))
        self.logo_unipv.setText(_fromUtf8(""))
        self.logo_unipv.setPixmap(QtGui.QPixmap(_fromUtf8(".sensum/cambridge.png")))
        self.logo_unipv.setObjectName(_fromUtf8("logo_unipv"))
        self.comboBox_index = QtGui.QComboBox(TemporalGraph)
        self.comboBox_index.setGeometry(QtCore.QRect(150, 270, 69, 22))
        self.comboBox_index.setObjectName(_fromUtf8("comboBox_index"))
        self.comboBox_index.addItem(_fromUtf8(""))
        self.comboBox_index.addItem(_fromUtf8(""))
        self.comboBox_index.addItem(_fromUtf8(""))
        self.comboBox_index.addItem(_fromUtf8(""))
        self.comboBox_index.addItem(_fromUtf8(""))
        self.comboBox_index.addItem(_fromUtf8(""))
        self.comboBox_index.addItem(_fromUtf8(""))
        self.comboBox_index.addItem(_fromUtf8(""))
        self.comboBox_index.addItem(_fromUtf8(""))
        self.comboBox_index.addItem(_fromUtf8(""))
        self.comboBox_index.addItem(_fromUtf8(""))
        self.comboBox_index.addItem(_fromUtf8(""))

        self.retranslateUi(TemporalGraph)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("accepted()")), TemporalGraph.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("rejected()")), TemporalGraph.reject)
        QtCore.QMetaObject.connectSlotsByName(TemporalGraph)

    def retranslateUi(self, TemporalGraph):
        TemporalGraph.setWindowTitle(_translate("TemporalGraph", "Dialog", None))
        self.pushButton_folder.setText(_translate("TemporalGraph", "...", None))
        self.label_title.setText(_translate("TemporalGraph", "<html><head/><body><p><span style=\" font-size:16pt;\">Temporal Analysis Graphs</span></p></body></html>", None))
        self.label_multiband.setText(_translate("TemporalGraph", "Input Landsat Images Folder", None))
        self.comboBox_index.setItemText(0, _translate("TemporalGraph", "Index1", None))
        self.comboBox_index.setItemText(1, _translate("TemporalGraph", "Index2", None))
        self.comboBox_index.setItemText(2, _translate("TemporalGraph", "Index3", None))
        self.comboBox_index.setItemText(3, _translate("TemporalGraph", "Index4", None))
        self.comboBox_index.setItemText(4, _translate("TemporalGraph", "Index5", None))
        self.comboBox_index.setItemText(5, _translate("TemporalGraph", "Index6", None))
        self.comboBox_index.setItemText(6, _translate("TemporalGraph", "Index7", None))
        self.comboBox_index.setItemText(7, _translate("TemporalGraph", "Index8", None))
        self.comboBox_index.setItemText(8, _translate("TemporalGraph", "Index9", None))
        self.comboBox_index.setItemText(9, _translate("TemporalGraph", "Index10", None))
        self.comboBox_index.setItemText(10, _translate("TemporalGraph", "Index11", None))
        self.comboBox_index.setItemText(11, _translate("TemporalGraph", "Index12", None))

