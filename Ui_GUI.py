# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\lz355\Documents\GitHub\Human_faces_Recognize\code\GUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(500, 350)
        MainWindow.setMinimumSize(QtCore.QSize(500, 350))
        MainWindow.setMaximumSize(QtCore.QSize(500, 350))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        MainWindow.setFont(font)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/images/fire.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.HeaderLine = QtWidgets.QFrame(self.centralwidget)
        self.HeaderLine.setGeometry(QtCore.QRect(0, 24, 500, 8))
        self.HeaderLine.setMinimumSize(QtCore.QSize(500, 8))
        self.HeaderLine.setMaximumSize(QtCore.QSize(500, 8))
        self.HeaderLine.setLineWidth(1)
        self.HeaderLine.setFrameShape(QtWidgets.QFrame.HLine)
        self.HeaderLine.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.HeaderLine.setObjectName("HeaderLine")
        self.HeaderText = QtWidgets.QLabel(self.centralwidget)
        self.HeaderText.setGeometry(QtCore.QRect(193, 0, 114, 26))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(10)
        self.HeaderText.setFont(font)
        self.HeaderText.setStyleSheet("color:#ffffff")
        self.HeaderText.setObjectName("HeaderText")
        self.ImgLabel = QtWidgets.QLabel(self.centralwidget)
        self.ImgLabel.setGeometry(QtCore.QRect(150, 40, 200, 200))
        self.ImgLabel.setMinimumSize(QtCore.QSize(200, 200))
        self.ImgLabel.setMaximumSize(QtCore.QSize(200, 200))
        self.ImgLabel.setAutoFillBackground(False)
        self.ImgLabel.setStyleSheet("")
        self.ImgLabel.setText("")
        self.ImgLabel.setObjectName("ImgLabel")
        self.ButtonGetResult = QtWidgets.QPushButton(self.centralwidget)
        self.ButtonGetResult.setGeometry(QtCore.QRect(400, 210, 80, 30))
        self.ButtonGetResult.setMinimumSize(QtCore.QSize(80, 30))
        self.ButtonGetResult.setMaximumSize(QtCore.QSize(80, 30))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(9)
        self.ButtonGetResult.setFont(font)
        self.ButtonGetResult.setStyleSheet("color: rgb(255, 255, 255);\n"
"background-color: rgb(85, 85, 127);")
        self.ButtonGetResult.setObjectName("ButtonGetResult")
        self.leftline = QtWidgets.QFrame(self.centralwidget)
        self.leftline.setGeometry(QtCore.QRect(376, 27, 8, 323))
        self.leftline.setMinimumSize(QtCore.QSize(8, 323))
        self.leftline.setMaximumSize(QtCore.QSize(8, 323))
        self.leftline.setFrameShape(QtWidgets.QFrame.VLine)
        self.leftline.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.leftline.setObjectName("leftline")
        self.rightline = QtWidgets.QFrame(self.centralwidget)
        self.rightline.setGeometry(QtCore.QRect(116, 27, 8, 323))
        self.rightline.setMinimumSize(QtCore.QSize(8, 323))
        self.rightline.setMaximumSize(QtCore.QSize(8, 323))
        self.rightline.setFrameShape(QtWidgets.QFrame.VLine)
        self.rightline.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.rightline.setObjectName("rightline")
        self.ButtonChooseFiles = QtWidgets.QPushButton(self.centralwidget)
        self.ButtonChooseFiles.setGeometry(QtCore.QRect(20, 290, 80, 30))
        self.ButtonChooseFiles.setMinimumSize(QtCore.QSize(80, 30))
        self.ButtonChooseFiles.setMaximumSize(QtCore.QSize(80, 30))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(9)
        self.ButtonChooseFiles.setFont(font)
        self.ButtonChooseFiles.setStyleSheet("background-color: rgb(85, 85, 127);\n"
"color: rgb(255, 255, 255);")
        self.ButtonChooseFiles.setObjectName("ButtonChooseFiles")
        self.SpaceBottom = QtWidgets.QListView(self.centralwidget)
        self.SpaceBottom.setGeometry(QtCore.QRect(0, 328, 500, 24))
        self.SpaceBottom.setMinimumSize(QtCore.QSize(500, 24))
        self.SpaceBottom.setMaximumSize(QtCore.QSize(500, 24))
        self.SpaceBottom.setStyleSheet("background:black;")
        self.SpaceBottom.setObjectName("SpaceBottom")
        self.ButtonQuitBottom = QtWidgets.QPushButton(self.centralwidget)
        self.ButtonQuitBottom.setGeometry(QtCore.QRect(400, 290, 80, 30))
        self.ButtonQuitBottom.setMinimumSize(QtCore.QSize(80, 30))
        self.ButtonQuitBottom.setMaximumSize(QtCore.QSize(80, 30))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(9)
        self.ButtonQuitBottom.setFont(font)
        self.ButtonQuitBottom.setStyleSheet("color: rgb(255, 255, 255);\n"
"background-color: rgb(85, 85, 127);")
        self.ButtonQuitBottom.setObjectName("ButtonQuitBottom")
        self.TextBottomleft = QtWidgets.QLabel(self.centralwidget)
        self.TextBottomleft.setGeometry(QtCore.QRect(0, 328, 130, 24))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        self.TextBottomleft.setFont(font)
        self.TextBottomleft.setStyleSheet("color:#ffffff")
        self.TextBottomleft.setObjectName("TextBottomleft")
        self.HeaderIcon = QtWidgets.QListView(self.centralwidget)
        self.HeaderIcon.setGeometry(QtCore.QRect(0, 0, 26, 26))
        self.HeaderIcon.setMinimumSize(QtCore.QSize(26, 26))
        self.HeaderIcon.setMaximumSize(QtCore.QSize(26, 26))
        self.HeaderIcon.setStyleSheet("border-image: url(:/images/fire.png);")
        self.HeaderIcon.setObjectName("HeaderIcon")
        self.HeaderSpace = QtWidgets.QListView(self.centralwidget)
        self.HeaderSpace.setGeometry(QtCore.QRect(0, 0, 500, 28))
        self.HeaderSpace.setMinimumSize(QtCore.QSize(500, 28))
        self.HeaderSpace.setMaximumSize(QtCore.QSize(500, 28))
        self.HeaderSpace.setStyleSheet("background:black;")
        self.HeaderSpace.setObjectName("HeaderSpace")
        self.ButtonMiniTop = QtWidgets.QPushButton(self.centralwidget)
        self.ButtonMiniTop.setGeometry(QtCore.QRect(435, 1, 26, 25))
        self.ButtonMiniTop.setMinimumSize(QtCore.QSize(26, 25))
        self.ButtonMiniTop.setMaximumSize(QtCore.QSize(26, 25))
        self.ButtonMiniTop.setStyleSheet("background-color: rgb(0, 0, 0);\n"
"border-image: url(:/images/heavy_minus_sign.png);")
        self.ButtonMiniTop.setText("")
        self.ButtonMiniTop.setObjectName("ButtonMiniTop")
        self.ButtonQuitTop = QtWidgets.QPushButton(self.centralwidget)
        self.ButtonQuitTop.setGeometry(QtCore.QRect(474, 1, 26, 25))
        self.ButtonQuitTop.setMinimumSize(QtCore.QSize(26, 25))
        self.ButtonQuitTop.setMaximumSize(QtCore.QSize(26, 25))
        self.ButtonQuitTop.setStyleSheet("background-color: rgb(0, 0, 0);\n"
"border-image: url(:/images/x.png);")
        self.ButtonQuitTop.setText("")
        self.ButtonQuitTop.setObjectName("ButtonQuitTop")
        self.labelImgName = QtWidgets.QLabel(self.centralwidget)
        self.labelImgName.setGeometry(QtCore.QRect(150, 240, 200, 24))
        self.labelImgName.setMinimumSize(QtCore.QSize(200, 24))
        self.labelImgName.setMaximumSize(QtCore.QSize(200, 24))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei")
        self.labelImgName.setFont(font)
        self.labelImgName.setStyleSheet("color: rgb(255, 255, 255);\n"
"background-color: rgba(170, 170, 255, 150);")
        self.labelImgName.setObjectName("labelImgName")
        self.labelExpGender = QtWidgets.QLabel(self.centralwidget)
        self.labelExpGender.setGeometry(QtCore.QRect(150, 264, 200, 24))
        self.labelExpGender.setMinimumSize(QtCore.QSize(200, 24))
        self.labelExpGender.setMaximumSize(QtCore.QSize(200, 24))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei")
        self.labelExpGender.setFont(font)
        self.labelExpGender.setStyleSheet("color: rgb(255, 255, 255);\n"
"background-color: rgba(170, 170, 255, 150);")
        self.labelExpGender.setObjectName("labelExpGender")
        self.labelThouGender = QtWidgets.QLabel(self.centralwidget)
        self.labelThouGender.setGeometry(QtCore.QRect(150, 288, 200, 24))
        self.labelThouGender.setMinimumSize(QtCore.QSize(200, 24))
        self.labelThouGender.setMaximumSize(QtCore.QSize(200, 24))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei")
        self.labelThouGender.setFont(font)
        self.labelThouGender.setStyleSheet("color: rgb(255, 255, 255);\n"
"background-color: rgba(170, 170, 255, 150);")
        self.labelThouGender.setObjectName("labelThouGender")
        self.groupBoxQuickChoose = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBoxQuickChoose.setGeometry(QtCore.QRect(10, 160, 100, 80))
        self.groupBoxQuickChoose.setMinimumSize(QtCore.QSize(100, 80))
        self.groupBoxQuickChoose.setMaximumSize(QtCore.QSize(100, 80))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei")
        self.groupBoxQuickChoose.setFont(font)
        self.groupBoxQuickChoose.setStyleSheet("QGroupBox{\n"
"color: rgb(255, 255, 255);\n"
"}")
        self.groupBoxQuickChoose.setObjectName("groupBoxQuickChoose")
        self.comboBoxGender = QtWidgets.QComboBox(self.groupBoxQuickChoose)
        self.comboBoxGender.setGeometry(QtCore.QRect(10, 20, 80, 20))
        self.comboBoxGender.setMinimumSize(QtCore.QSize(80, 20))
        self.comboBoxGender.setMaximumSize(QtCore.QSize(80, 20))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei")
        self.comboBoxGender.setFont(font)
        self.comboBoxGender.setStyleSheet("color: rgb(255, 255, 255);\n"
"background-color: rgb(85, 85, 127);")
        self.comboBoxGender.setObjectName("comboBoxGender")
        self.comboBoxGender.addItem("")
        self.comboBoxGender.addItem("")
        self.comboBoxPath = QtWidgets.QComboBox(self.groupBoxQuickChoose)
        self.comboBoxPath.setGeometry(QtCore.QRect(10, 50, 80, 20))
        self.comboBoxPath.setMinimumSize(QtCore.QSize(80, 20))
        self.comboBoxPath.setMaximumSize(QtCore.QSize(80, 20))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei")
        self.comboBoxPath.setFont(font)
        self.comboBoxPath.setStyleSheet("color: rgb(255, 255, 255);\n"
"background-color: rgb(85, 85, 127);")
        self.comboBoxPath.setObjectName("comboBoxPath")
        self.TextBottomright = QtWidgets.QLabel(self.centralwidget)
        self.TextBottomright.setGeometry(QtCore.QRect(370, 326, 130, 24))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        self.TextBottomright.setFont(font)
        self.TextBottomright.setStyleSheet("color:#ffffff")
        self.TextBottomright.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.TextBottomright.setObjectName("TextBottomright")
        self.listViewBgImg = QtWidgets.QListView(self.centralwidget)
        self.listViewBgImg.setGeometry(QtCore.QRect(0, 27, 500, 302))
        self.listViewBgImg.setStyleSheet("border-image: url(:/images/milky.jpg);")
        self.listViewBgImg.setObjectName("listViewBgImg")
        self.labelImpleRight = QtWidgets.QLabel(self.centralwidget)
        self.labelImpleRight.setGeometry(QtCore.QRect(307, 0, 128, 26))
        self.labelImpleRight.setObjectName("labelImpleRight")
        self.labelImpleLeft = QtWidgets.QLabel(self.centralwidget)
        self.labelImpleLeft.setGeometry(QtCore.QRect(26, 0, 167, 26))
        self.labelImpleLeft.setObjectName("labelImpleLeft")
        self.line_h1 = QtWidgets.QFrame(self.centralwidget)
        self.line_h1.setGeometry(QtCore.QRect(150, 262, 200, 4))
        self.line_h1.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_h1.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_h1.setObjectName("line_h1")
        self.line_h2 = QtWidgets.QFrame(self.centralwidget)
        self.line_h2.setGeometry(QtCore.QRect(150, 286, 200, 4))
        self.line_h2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_h2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_h2.setObjectName("line_h2")
        self.line_h3 = QtWidgets.QFrame(self.centralwidget)
        self.line_h3.setGeometry(QtCore.QRect(150, 238, 200, 4))
        self.line_h3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_h3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_h3.setObjectName("line_h3")
        self.line_h4 = QtWidgets.QFrame(self.centralwidget)
        self.line_h4.setGeometry(QtCore.QRect(150, 310, 200, 4))
        self.line_h4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_h4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_h4.setObjectName("line_h4")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(349, 240, 2, 72))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(149, 240, 2, 72))
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.listViewBgImg.raise_()
        self.HeaderSpace.raise_()
        self.HeaderLine.raise_()
        self.HeaderText.raise_()
        self.ImgLabel.raise_()
        self.ButtonGetResult.raise_()
        self.leftline.raise_()
        self.rightline.raise_()
        self.ButtonChooseFiles.raise_()
        self.SpaceBottom.raise_()
        self.ButtonQuitBottom.raise_()
        self.TextBottomleft.raise_()
        self.HeaderIcon.raise_()
        self.ButtonMiniTop.raise_()
        self.ButtonQuitTop.raise_()
        self.labelImgName.raise_()
        self.labelExpGender.raise_()
        self.labelThouGender.raise_()
        self.groupBoxQuickChoose.raise_()
        self.TextBottomright.raise_()
        self.labelImpleRight.raise_()
        self.labelImpleLeft.raise_()
        self.line_h1.raise_()
        self.line_h2.raise_()
        self.line_h3.raise_()
        self.line_h4.raise_()
        self.line.raise_()
        self.line_2.raise_()
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.ButtonQuitBottom.clicked.connect(MainWindow.close)
        self.ButtonQuitTop.clicked.connect(MainWindow.close)
        self.ButtonMiniTop.clicked.connect(MainWindow.showMinimized)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Facial Recognization"))
        self.HeaderText.setText(_translate("MainWindow", "Facial Recognition"))
        self.ButtonGetResult.setText(_translate("MainWindow", "Get Results"))
        self.ButtonChooseFiles.setText(_translate("MainWindow", "Image-Files"))
        self.ButtonQuitBottom.setText(_translate("MainWindow", "Exit"))
        self.TextBottomleft.setText(_translate("MainWindow", "??2021 Zsbyqx-THUEE"))
        self.labelImgName.setText(_translate("MainWindow", "Name:"))
        self.labelExpGender.setText(_translate("MainWindow", "Expected Gender:"))
        self.labelThouGender.setText(_translate("MainWindow", "Predicted Gender:"))
        self.groupBoxQuickChoose.setTitle(_translate("MainWindow", "Quick-Choose"))
        self.comboBoxGender.setItemText(0, _translate("MainWindow", "Female"))
        self.comboBoxGender.setItemText(1, _translate("MainWindow", "Male"))
        self.TextBottomright.setText(_translate("MainWindow", "Python  "))
        self.labelImpleRight.setText(_translate("MainWindow", "TextLabel"))
        self.labelImpleLeft.setText(_translate("MainWindow", "TextLabel"))
import images_rc
