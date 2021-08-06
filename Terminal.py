from Windows import*
import sys
from PyQt5.QtGui import*
from PyQt5.QtWidgets import *
from PyQt5.QtCore import*
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setStyle(QStyleFactory.create('fusion'))
    
    app = QApplication(sys.argv)
    
    win = WinMain()
    win.show()
    
    sys.exit(app.exec_())