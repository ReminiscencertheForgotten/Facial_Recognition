from PyQt5.QtGui import*
from PyQt5.QtWidgets import *
from PyQt5.QtCore import*
import os, torch
from Ui_GUI import*
from network import*
from PIL import Image
import torchvision.transforms as transforms

network = Network()
network.eval()
network.load_state_dict(torch.load('TrainResult/Model0.95.pth'))

class WinMain(Ui_MainWindow, QMainWindow):
    
    def __init__(self, diag=None):
        super().__init__()
        self.diag = diag
        self.NameListPrePare()
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.combine()

    def NameListPrePare(self):
        ffp = open('female_names.txt', 'r');mfp = open('male_names.txt', 'r')
        female = ffp.readlines();male = mfp.readlines()
        self.femaleNames = [];self.maleNames = []
        for ele in female:
            self.femaleNames.append(ele[:-1])
        for ele in male:
            self.maleNames.append(ele[:-1])
        self.femaleNames.sort();self.maleNames.sort()
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.m_flag = True
            self.m_Position = event.globalPos() - self.pos()
            event.accept()
            self.setCursor(QCursor(Qt.OpenHandCursor))
 
    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)
            QMouseEvent.accept()
 
    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False
        self.setCursor(QCursor(Qt.ArrowCursor))
       
    def combine(self):
        super().setupUi(self)
        self.initComboBox()
        self.setImage()
        self.comboBoxGender.currentIndexChanged.connect(self.initComboBox)
        self.comboBoxPath.currentIndexChanged.connect(self.setImage)
        self.ButtonGetResult.clicked.connect(self.getResult)
        self.ButtonChooseFiles.clicked.connect(self.fileChoose)
        
    def initComboBox(self):
        gender = self.comboBoxGender.currentText()
        self.comboBoxPath.clear()
        if gender == 'Female':
            self.comboBoxPath.addItems(self.femaleNames)
        else:
            self.comboBoxPath.addItems(self.maleNames)
            
    def setImage(self):
        if self.comboBoxPath.currentText():
            if self.groupBoxQuickChoose.isEnabled():
                self.gender = self.comboBoxGender.currentText()
                self.path = self.comboBoxPath.currentText()
            self.labelExpGender.setText('Expected Gender: ' + self.gender)
            ownerName = self.path[:-9]
            self.labelImgName.setText('Name: ' + ownerName)
            imgPath = os.path.join('Dataset', ownerName, self.path)
            self.ImgLabel.setPixmap(QPixmap(imgPath))
            self.ImgLabel.setScaledContents(True)
            self.labelThouGender.setText('Predicted Gender: ')
            self.signalFlag = 0
        
    
    def fileChoose(self):
        self.groupBoxQuickChoose.setEnabled(False)
        cwd = os.getcwd() + '\\Dataset'
        fileName, _ = QFileDialog.getOpenFileName(
            self, 'Choose Images from all files', cwd)
        self.path = fileName.split('/')[-1];
        ownerName = fileName.split('/')[-2];
        if ownerName in self.femaleNames:
            self.gender = 'Female'
        else:
            self.gender = 'Male'
        self.setImage()

    def getResult(self):
        ownerName = self.path[:-9]
        self.labelImgName.setText('Name: ' + ownerName)
        imgPath = os.path.join('Dataset', ownerName, self.path)
        image = Image.open(imgPath).crop((60, 55, 190, 190))
        transformer_normal = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(125),
            transforms.RandomCrop(125, padding=15),
            transforms.Normalize(
                mean=[0.471, 0.448, 0.408], 
                std=[0.234, 0.239, 0.242]
            )       
        ])
        image = transformer_normal(image).unsqueeze(0)
        with torch.no_grad():
            pred = network(image)
        
        relist = ['Female', 'Male']
        _, predicted = torch.max(pred, 1)
        
        gen = relist[predicted[0]]
        self.labelThouGender.setText('Predicted Gender: ' + gen)
        self.groupBoxQuickChoose.setEnabled(True)
