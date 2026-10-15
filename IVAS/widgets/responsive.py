"""Screen-aware containers for desktop tools with large or translated forms."""

from PyQt5 import QtCore, QtWidgets


class CurrentWorkspaceStack(QtWidgets.QStackedWidget):
    """Hidden editors must not impose their minimum height on another mode."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.currentChanged.connect(self.updateGeometry)

    def minimumSizeHint(self):
        widget = self.currentWidget()
        return widget.minimumSizeHint() if widget is not None else QtCore.QSize(0, 0)

    def sizeHint(self):
        widget = self.currentWidget()
        return widget.sizeHint() if widget is not None else QtCore.QSize(0, 0)


def scrollable_content(content, parent=None):
    """Keep a form's natural minimum size reachable without enlarging its host."""
    scroll = QtWidgets.QScrollArea(parent)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
    scroll.setMinimumSize(0, 0)
    if content.layout() is not None:
        content.layout().setSizeConstraint(QtWidgets.QLayout.SetMinAndMaxSize)
    scroll.setWidget(content)
    return scroll


class ResponsiveDialog(QtWidgets.QDialog):
    """Retain Designer forms, keeping their footer and overflow accessible."""

    def _prepare_responsive_layout(self):
        if getattr(self, "_responsive_scroll", None) is not None or self.layout() is None:
            return
        original = self.layout()
        preferred = self.size()
        footer = None
        # Designer dialogs use a footer widget/layout; hand-written forms use
        # QDialogButtonBox. Keep either outside the scrollable form.
        for index in range(original.count() - 1, -1, -1):
            item = original.itemAt(index)
            if item.spacerItem() is not None:
                continue
            if self._is_footer(item):
                footer = original.takeAt(index)
            break
        content = QtWidgets.QWidget()
        content.setObjectName("responsiveDialogContent")
        content.setLayout(original)
        self._responsive_scroll = scrollable_content(content, self)
        self._responsive_scroll.setObjectName("responsiveDialogScroll")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self._responsive_scroll, 1)
        if footer is not None:
            if footer.widget() is not None:
                layout.addWidget(footer.widget())
            else:
                footer.layout().setParent(None)
                layout.addLayout(footer.layout())
        self.setMinimumSize(0, 0)
        self.setMaximumSize(QtWidgets.QWIDGETSIZE_MAX, QtWidgets.QWIDGETSIZE_MAX)
        self.resize(preferred)

    @classmethod
    def _is_footer(cls, item):
        widget = item.widget()
        if widget is not None:
            if isinstance(widget, QtWidgets.QDialogButtonBox):
                return True
            controls = [widget] + widget.findChildren(QtWidgets.QWidget)
            return (any(control.objectName() == "pushButton_close" for control in controls)
                    and not any(isinstance(control, (QtWidgets.QLabel, QtWidgets.QLineEdit,
                                                      QtWidgets.QAbstractItemView))
                                for control in controls))
        layout = item.layout()
        return bool(layout and any(cls._is_footer(layout.itemAt(i)) for i in range(layout.count())))

    def _fit_available_screen(self, *_args):
        screen = self.screen()
        if screen is None:
            return
        available = screen.availableGeometry().adjusted(12, 12, -12, -12)
        decoration = self.frameGeometry().size() - self.size()
        self.resize(self.size().boundedTo(available.size() - decoration))
        frame = self.frameGeometry()
        x = max(available.left(), min(frame.left(), available.right() - frame.width() + 1))
        y = max(available.top(), min(frame.top(), available.bottom() - frame.height() + 1))
        self.move(self.pos() + QtCore.QPoint(x - frame.left(), y - frame.top()))

    def showEvent(self, event):
        self._prepare_responsive_layout()
        super().showEvent(event)
        self._fit_available_screen()
        handle = self.windowHandle()
        if handle is not None and not getattr(self, "_screen_connected", False):
            handle.screenChanged.connect(self._fit_available_screen)
            self._screen_connected = True
