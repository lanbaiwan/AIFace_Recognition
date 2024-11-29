# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'form.ui'
##
## Created by: Qt User Interface Compiler version 6.8.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QLayout,
    QLineEdit, QPushButton, QSizePolicy, QSpacerItem,
    QVBoxLayout, QWidget)

class Ui_AIface(object):
    def setupUi(self, AIface):
        if not AIface.objectName():
            AIface.setObjectName(u"AIface")
        AIface.resize(800, 600)
        self.verticalLayoutWidget = QWidget(AIface)
        self.verticalLayoutWidget.setObjectName(u"verticalLayoutWidget")
        self.verticalLayoutWidget.setGeometry(QRect(160, 50, 481, 508))
        self.verticalLayout = QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setSpacing(20)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setSizeConstraint(QLayout.SetMinimumSize)
        self.verticalLayout.setContentsMargins(20, 20, 20, 20)
        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.imageLabel = QLabel(self.verticalLayoutWidget)
        self.imageLabel.setObjectName(u"imageLabel")
        self.imageLabel.setMinimumSize(QSize(300, 300))
        self.imageLabel.setMaximumSize(QSize(300, 300))

        self.horizontalLayout_3.addWidget(self.imageLabel)


        self.verticalLayout.addLayout(self.horizontalLayout_3)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setSizeConstraint(QLayout.SetMinimumSize)
        self.horizontalLayout_2.setContentsMargins(20, 20, 20, 20)
        self.selectButton = QPushButton(self.verticalLayoutWidget)
        self.selectButton.setObjectName(u"selectButton")

        self.horizontalLayout_2.addWidget(self.selectButton)

        self.horizontalSpacer = QSpacerItem(60, 20, QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer)

        self.confirmButton = QPushButton(self.verticalLayoutWidget)
        self.confirmButton.setObjectName(u"confirmButton")

        self.horizontalLayout_2.addWidget(self.confirmButton)


        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setSpacing(6)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.horizontalLayout.setContentsMargins(20, 20, 20, 20)
        self.introLabel = QLabel(self.verticalLayoutWidget)
        self.introLabel.setObjectName(u"introLabel")

        self.horizontalLayout.addWidget(self.introLabel)

        self.validationLine = QLineEdit(self.verticalLayoutWidget)
        self.validationLine.setObjectName(u"validationLine")

        self.horizontalLayout.addWidget(self.validationLine)


        self.verticalLayout.addLayout(self.horizontalLayout)


        self.retranslateUi(AIface)

        QMetaObject.connectSlotsByName(AIface)
    # setupUi

    def retranslateUi(self, AIface):
        AIface.setWindowTitle(QCoreApplication.translate("AIface", u"Widget", None))
        self.imageLabel.setText("")
        self.selectButton.setText(QCoreApplication.translate("AIface", u"\u9009\u62e9", None))
        self.confirmButton.setText(QCoreApplication.translate("AIface", u"\u6d4b\u8bd5", None))
        self.introLabel.setText(QCoreApplication.translate("AIface", u"\u8be5\u56fe\u7247\u7684\u7f6e\u4fe1\u5ea6\u4e3a\uff1a", None))
    # retranslateUi

