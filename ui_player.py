# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'player.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1100, 658)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.videoWidget = QtWidgets.QWidget(self.centralwidget)
        self.videoWidget.setEnabled(True)
        self.videoWidget.setGeometry(QtCore.QRect(20, 10, 640, 360))
        self.videoWidget.setObjectName("videoWidget")
        self.MainChatwidget = QtWidgets.QWidget(self.centralwidget)
        self.MainChatwidget.setGeometry(QtCore.QRect(20, 390, 641, 211))
        self.MainChatwidget.setObjectName("MainChatwidget")
        self.widget_2 = QtWidgets.QWidget(self.centralwidget)
        self.widget_2.setGeometry(QtCore.QRect(680, 10, 401, 171))
        self.widget_2.setObjectName("widget_2")
        self.Chatwidget_1 = QtWidgets.QWidget(self.widget_2)
        self.Chatwidget_1.setGeometry(QtCore.QRect(0, 0, 401, 171))
        self.Chatwidget_1.setObjectName("Chatwidget_1")
        self.Chatwidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.Chatwidget_2.setGeometry(QtCore.QRect(680, 200, 401, 171))
        self.Chatwidget_2.setObjectName("Chatwidget_2")
        self.Chatwidget_3 = QtWidgets.QWidget(self.centralwidget)
        self.Chatwidget_3.setGeometry(QtCore.QRect(680, 390, 401, 211))
        self.Chatwidget_3.setObjectName("Chatwidget_3")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1100, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
