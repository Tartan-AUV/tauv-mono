from python_qt_binding.QtWidgets import QWidget
from python_qt_binding.QtCore import Signal, QObject, Slot, QRect, QLine, Qt, QPoint
from python_qt_binding.QtGui import QPainter, QColor, QPen

from joy_model import JoyModel

class AxisWidget(QWidget):
    def __init__(self, parent=None):
        super(AxisWidget, self).__init__(parent)
        super(AxisWidget, self).setMinimumSize(100, 100)
        self._x = 0
        self._y = 0

    @Slot(float)
    def setX(self, x):
        self._x = x
        super(AxisWidget, self).update()

    @Slot(float)
    def setY(self, y):
        self._y = y
        super(AxisWidget, self).update()

    @Slot(float,float)
    def setCoords(self, x, y):
        self._x = x
        self._y = y
        super(AxisWidget, self).update()

    def paintEvent(self, event):
        geom = super(AxisWidget, self).geometry()
        h = geom.height()
        w = geom.width()


        box = QRect(2, 2, w-4, h-4)
        horiz = QLine(2, h/2, w-2, h/2)
        vert = QLine(w/2, 2, w/2, h-2)
        targ = QPoint(self._x*(w-4)/2+w/2, self._y*(h-4)/2+h/2)

        plt = super(AxisWidget, self).palette()
        linebrsh = plt.dark()
        targetbrsh = plt.highlight()

        linepen = QPen(linebrsh, 1, Qt.SolidLine, Qt.SquareCap)
        targetpen = QPen(targetbrsh, 2, Qt.SolidLine, Qt.SquareCap)

        qp = QPainter()
        qp.begin(self)
        qp.setPen(linepen)
        qp.drawRect(box)
        qp.drawLine(horiz)
        qp.drawLine(vert)
        qp.setPen(targetpen)
        qp.drawEllipse(targ, 10, 10)
        qp.end()
