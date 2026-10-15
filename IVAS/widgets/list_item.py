"""Size embedded list rows from their styled controls instead of pixel constants."""

from PyQt5 import QtWidgets


def fit_list_item(item, widget):
    view = widget.parentWidget()
    while view is not None and not isinstance(view, QtWidgets.QListWidget):
        view = view.parentWidget()
    if view is not None and not view.property("embeddedRowSpacing"):
        # The row layout owns spacing. Delegate padding would otherwise be
        # subtracted from SizeHintRole and clip the embedded controls.
        view.setProperty("embeddedRowSpacing", True)
        view.setStyleSheet(view.styleSheet() + "\nQListWidget::item { padding: 0px; }")
    widget.ensurePolished()
    for child in widget.findChildren(QtWidgets.QWidget):
        child.ensurePolished()
    widget.layout().invalidate()
    widget.layout().activate()
    item.setSizeHint(widget.sizeHint().expandedTo(widget.minimumSizeHint()))
