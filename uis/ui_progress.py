# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'uis/ui_progress.ui'
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

class Ui_Progress(object):
    def setupUi(self, Progress):
        Progress.setObjectName(_fromUtf8("Progress"))
        Progress.resize(568, 119)
        Progress.setMinimumSize(QtCore.QSize(568, 119))
        Progress.setMaximumSize(QtCore.QSize(568, 119))
        self.label_title = QtGui.QLabel(Progress)
        self.label_title.setGeometry(QtCore.QRect(2, 8, 551, 21))
        self.label_title.setObjectName(_fromUtf8("label_title"))
        self.progressBar = QtGui.QProgressBar(Progress)
        self.progressBar.setGeometry(QtCore.QRect(30, 50, 521, 41))
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName(_fromUtf8("progressBar"))

        self.retranslateUi(Progress)
        QtCore.QMetaObject.connectSlotsByName(Progress)

    def retranslateUi(self, Progress):
        Progress.setWindowTitle(_translate("Progress", "SensumTools", None))
        self.label_title.setText(_translate("Progress", "<html><head/><body><p align=\"center\"><span style=\" font-size:16pt;\">Please wait...</span></p></body></html>", None))

