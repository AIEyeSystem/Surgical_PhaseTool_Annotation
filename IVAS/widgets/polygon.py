# -*- coding: utf-8 -*-
# @Maintainer: zack zhao

import math
import typing

from PyQt5 import QtCore, QtGui, QtWidgets

from IVAS.annotation import Object, rectangle_segmentation
from IVAS.configs import STATUSMode


# ============================================================
#  Prompt point (SAM point prompt visual)
# ============================================================

class PromptPoint(QtWidgets.QGraphicsPathItem):
    """SAM prompt point."""

    def __init__(self, pos, type=0):
        super(PromptPoint, self).__init__()
        self.color = QtGui.QColor("#0000FF") if type == 0 else QtGui.QColor("#00FF00")
        self.color.setAlpha(255)
        self.painterpath = QtGui.QPainterPath()
        self.painterpath.addEllipse(QtCore.QRectF(-1, -1, 2, 2))
        self.setPath(self.painterpath)
        self.setBrush(self.color)
        self.setPen(QtGui.QPen(self.color, 3))
        self.setZValue(1e5)

        self.setPos(pos)


# ============================================================
#  Vertex hierarchy
# ============================================================

class BaseVertex(QtWidgets.QGraphicsPathItem):
    """Base class for all draggable handle vertices.

    Provides common appearance (ellipse shape, color, brush/pen),
    scene-boundary clamping, and selection-highlight behavior.

    Subclasses:
        PolygonVertex — selectable, with hover effects for polygon editing.
        LineVertex    — non-selectable, for repaint guide line.
        PromptRectVertex    — non-selectable, for SAM box prompt.
    """

    def __init__(self, parent_shape, color, nohover_size=2, selectable=True):
        super().__init__()
        self.parent_shape = parent_shape
        self.color = QtGui.QColor(color)
        self.color.setAlpha(255)
        self.nohover_size = nohover_size
        self.hover_size = self.nohover_size + 2
        self.line_width = 0

        self.nohover_path = self._make_ellipse(self.nohover_size)
        self.hover_path = self._make_ellipse(self.hover_size)

        self.setPath(self.nohover_path)
        self.setBrush(self.color)
        self.setPen(QtGui.QPen(self.color, self.line_width))
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, selectable)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True
        )
        self.setAcceptHoverEvents(True)
        self.setZValue(1e5)

    @staticmethod
    def _make_ellipse(size):
        """Create a circle-shaped QPainterPath centred at origin."""
        path = QtGui.QPainterPath()
        path.addEllipse(QtCore.QRectF(-size // 2, -size // 2, size, size))
        return path

    def setColor(self, color):
        """Update the vertex colour."""
        self.color = QtGui.QColor(color)
        self.color.setAlpha(255)
        self.setPen(QtGui.QPen(self.color, self.line_width))
        self.setBrush(self.color)

    def itemChange(
        self, change: "QtWidgets.QGraphicsItem.GraphicsItemChange", value: typing.Any
    ):
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self.scene().mainwindow.actionDelete.setEnabled(self.isSelected())
            if self.isSelected():
                self.setBrush(QtGui.QColor("#00A0FF"))
            else:
                self.color.setAlpha(255)
                self.setBrush(self.color)

        if (
            change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionChange
            and self.isEnabled()
        ):
            value = self._clamp_to_scene(value)
            index = self.parent_shape.vertices.index(self)
            self.parent_shape.movePoint(index, value)

        return super().itemChange(change, value)

    def _clamp_to_scene(self, value):
        """Constrain the vertex position to lie within the scene bounds."""
        if value.x() < 0:
            value.setX(0)
        if value.x() > self.scene().width() - 1:
            value.setX(self.scene().width() - 1)
        if value.y() < 0:
            value.setY(0)
        if value.y() > self.scene().height() - 1:
            value.setY(self.scene().height() - 1)
        return value


class PolygonVertex(BaseVertex):
    """Vertex for polygon annotation — selectable with hover effects."""

    def __init__(self, parent_shape, color, nohover_size=2):
        super().__init__(parent_shape, color, nohover_size, selectable=True)

    def hoverEnterEvent(self, event: "QGraphicsSceneHoverEvent"):
        self.scene().hovered_vertex = self
        if self.scene().mode == STATUSMode.CREATE:
            self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))
        else:  # EDIT, VIEW
            self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.OpenHandCursor))
            if not self.isSelected():
                self.setBrush(QtGui.QColor(255, 255, 255, 255))
            self.setPath(self.hover_path)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event: "QGraphicsSceneHoverEvent"):
        self.scene().hovered_vertex = None
        if not self.isSelected():
            self.color.setAlpha(255)
            self.setBrush(self.color)
        self.setPath(self.nohover_path)
        super().hoverLeaveEvent(event)


class LineVertex(BaseVertex):
    """Vertex for repaint guide line — non-selectable."""

    def __init__(self, parent_shape, color, nohover_size=2):
        super().__init__(parent_shape, color, nohover_size, selectable=False)


class PromptRectVertex(BaseVertex):
    """Vertex for SAM prompt rectangle — non-selectable."""

    def __init__(self, parent_shape, color, nohover_size=2):
        super().__init__(parent_shape, color, nohover_size, selectable=False)


class OBBVertex(PolygonVertex):
    """Vertex for OBB annotation — selectable with hover effects.

    Currently identical to PolygonVertex; exists as a dedicated type so that
    OBB-specific vertex behaviour (e.g. rotation handle) can be added later
    without affecting Polygon vertices.
    """

    def __init__(self, parent_shape, color, nohover_size=2):
        super().__init__(parent_shape, color, nohover_size)


# ============================================================
#  Shape mixin — points / vertices CRUD shared by Polygon, Line, PromptRect
# ============================================================

class BaseShape:
    """Mixin providing point-list and vertex management for shapes.

    Concrete classes must:
      1. Inherit a QGraphicsItem subclass *and* BaseShape.
      2. Call ``self._init_shape(vertex_cls)`` in ``__init__``.
      3. Implement ``redraw()`` to re-render the shape from ``self.points``.
    """

    def _init_shape(self, vertex_cls):
        """Initialise the point list and vertex factory.

        Arguments:
            vertex_cls: A BaseVertex subclass used to create draggable handles.
        """
        self.points: list = []
        self.vertices: list = []
        self._vertex_cls = vertex_cls
        self.origin_source = ""

    def mark_edited(self, notify=True):
        scene = self.scene()
        if scene is None or getattr(self, "is_drawing", True) or not scene.mainwindow.load_finished:
            return
        if getattr(self, "source", "manual") != "manual":
            self.origin_source = self.origin_source or self.source
            self.source = "manual"
        self.review_status = "draft"
        self.area = self.calculate_area()
        if notify:
            scene.mainwindow.set_saved_state(False)

    # ---- point / vertex CRUD ------------------------------------------------

    def addPoint(self, point: QtCore.QPointF):
        """Append a point and its corresponding visual vertex to the scene."""
        self.points.append(point)
        vertex_size = self.scene().mainwindow.cfg["software"]["vertex_size"] * 2
        vertex = self._vertex_cls(self, self.color, vertex_size)
        self.scene().addItem(vertex)
        self.vertices.append(vertex)
        vertex.setPos(point)

        if not getattr(self, "is_drawing", True):
            self.mark_edited()

    def movePoint(self, index: int, point: QtCore.QPointF):
        """Move the *index*-th point to a new scene position.

        Calls ``redraw()`` and the ``_on_point_moved`` hook so subclasses
        (e.g. Polygon) can add extra behaviour like real-time area updates.
        """
        if not 0 <= index < len(self.points):
            return
        self.points[index] = self.mapFromScene(point)
        self.redraw()
        self._on_point_moved(index, point)

    def _on_point_moved(self, index: int, point: QtCore.QPointF):
        """Hook invoked after every successful ``movePoint``.

        Override in subclasses that need side-effects (area calculation,
        dirty-state tracking, etc.).  The default implementation is a no-op.
        """

    def removePoint(self, index):
        """Remove the *index*-th point and its vertex.  Returns the removed point."""
        if not self.points:
            return None
        point = self.points.pop(index)
        vertex = self.vertices.pop(index)
        self.scene().removeItem(vertex)
        del vertex
        self.redraw()
        self.mark_edited()
        return point

    def delete(self):
        """Remove all points and vertices (e.g. when discarding the shape)."""
        self.points.clear()
        while self.vertices:
            vertex = self.vertices.pop()
            self.scene().removeItem(vertex)
            del vertex


# ============================================================
#  Polygon — full annotation shape
# ============================================================

class Polygon(QtWidgets.QGraphicsPolygonItem, BaseShape):
    """
    Polygon annotation.

    Attributes:
        line_width (int): The width of the edge.
        hover_alpha (int): The alpha value of the polygon when hovering.
        nohover_alpha (int): the alpha value of the polygon when nohovering.
        points (list): Record the point pos of the polygon.
        vertices (list[PolygonVertex]): Record the vertices of the polygon.
        is_drawing (bool): The flag to indicate if the polygon is drawing.

        category (str): The category of the polygon.
        group (int): The group of the polygon.
        iscrowd (bool): The flag to indicate if the polygon is crowd.
        note (str): The note of the polygon.
        area (float): The area of the polygon.
    """

    def __init__(self):
        QtWidgets.QGraphicsPolygonItem.__init__(self, parent=None)
        self._init_shape(PolygonVertex)

        self.line_width = 1
        self.hover_alpha = 150
        self.nohover_alpha = 80
        self.category = ""
        self.group = 0
        self.iscrowd = False
        self.note = ""
        self.object_id = ""
        self.track_id = ""
        self.source = "manual"
        self.review_status = "reviewed"
        self.area = 0

        self.color = QtGui.QColor("#ff0000")
        self.is_drawing = True
        pen = QtGui.QPen(self.color, self.line_width)
        pen.setStyle(QtCore.Qt.PenStyle.DotLine)
        self.setPen(pen)
        self.setBrush(QtGui.QBrush(self.color, QtCore.Qt.BrushStyle.FDiagPattern))

        self.setAcceptHoverEvents(True)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True
        )
        self.setZValue(1e5)

    # ---- point-moved hook ----------------------------------------------------

    def _on_point_moved(self, index: int, point: QtCore.QPointF):
        """Polygon-specific side-effects after a vertex is dragged."""
        if self.scene().mainwindow.cfg["software"]["real_time_area"]:
            self.area = self.calculate_area()
        if (
            self.scene().mainwindow.load_finished
            and not self.is_drawing
            and self.scene().mode != STATUSMode.REPAINT
        ):
            self.mark_edited()

    # ---- vertex-only move (used when the whole polygon is dragged) -----------

    def moveVertex(self, index, point):
        """
        Move the vertex at the given index to the given point.
        The vertex position is updated directly without going through movePoint.
        """
        if not 0 <= index < len(self.vertices):
            return
        vertex = self.vertices[index]
        vertex.setEnabled(False)
        vertex.setPos(point)
        vertex.setEnabled(True)

    # ---- Qt item events ------------------------------------------------------

    def itemChange(
        self, change: "QGraphicsItem.GraphicsItemChange", value: typing.Any
    ):
        if (
            change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged
            and not self.is_drawing
            and self.scene().mode != STATUSMode.CREATE
        ):
            if self.isSelected():
                color = QtGui.QColor("#00A0FF")
                color.setAlpha(self.hover_alpha)
                self.setBrush(color)
                self.scene().selected_polygons_list.append(self)
            else:
                self.color.setAlpha(self.nohover_alpha)
                self.setBrush(self.color)
                if self in self.scene().selected_polygons_list:
                    self.scene().selected_polygons_list.remove(self)
            self.scene().mainwindow.annos_dock_widget.set_selected(self)

        if (
            change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionChange
        ):
            if self.is_drawing:
                value = 0
            else:
                bias = value
                l, t, b, r = (
                    self.boundingRect().left(),
                    self.boundingRect().top(),
                    self.boundingRect().bottom(),
                    self.boundingRect().right(),
                )
                if l + bias.x() < 0:
                    bias.setX(-l)
                if r + bias.x() > self.scene().width():
                    bias.setX(self.scene().width() - r)
                if t + bias.y() < 0:
                    bias.setY(-t)
                if b + bias.y() > self.scene().height():
                    bias.setY(self.scene().height() - b)

                for index, point in enumerate(self.points):
                    self.moveVertex(index, point + bias)

                if self.scene().mainwindow.load_finished:
                    self.mark_edited()

        if (
            change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged
            and self.isSelected()
        ):
            self.setSelected(not self.is_drawing)
        return super().itemChange(change, value)

    def hoverEnterEvent(self, event: "QGraphicsSceneHoverEvent"):
        if not self.is_drawing and not self.isSelected():
            self.color.setAlpha(self.hover_alpha)
            self.setBrush(self.color)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event: "QGraphicsSceneHoverEvent"):
        if not self.is_drawing and not self.isSelected():
            self.color.setAlpha(self.nohover_alpha)
            self.setBrush(self.color)
        super().hoverLeaveEvent(event)

    def mouseDoubleClickEvent(self, event: "QGraphicsSceneMouseEvent"):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            editor = self.scene().mainwindow._ensure_category_edit_dialog()
            editor.polygons = [self]
            editor.load_cfg()
            editor.show()

    # ---- rendering -----------------------------------------------------------

    def redraw(self):
        if len(self.points) < 1:
            return
        self.setPolygon(QtGui.QPolygonF(self.points))

    def change_color(self, color: QtGui.QColor):
        self.color = color
        if not self.scene().mainwindow.cfg["software"]["show_edge"]:
            color.setAlpha(0)
        self.setPen(QtGui.QPen(color, self.line_width))
        self.color.setAlpha(self.nohover_alpha)
        self.setBrush(self.color)

        vertex_color = self.color
        vertex_color.setAlpha(255)
        for vertex in self.vertices:
            vertex.setPen(QtGui.QPen(vertex_color, self.line_width))
            vertex.setBrush(vertex_color)

    # ---- lifecycle -----------------------------------------------------------

    def set_drawed(
        self,
        category: str,
        group: int,
        iscrowd: bool,
        note: str,
        color: QtGui.QColor,
        layer: int = None,
    ):
        """
        Set attributes for polygon and set is_drawing attribute to False.
        """
        self.is_drawing = False
        self.category = category
        if isinstance(group, str):
            group = 0 if group == "" else int(group)
        self.group = group
        self.iscrowd = iscrowd
        self.note = note

        self.color = color
        self.color.setAlpha(255)

        if not self.scene().mainwindow.cfg["software"]["show_edge"]:
            self.color.setAlpha(0)
        self.setPen(QtGui.QPen(self.color, self.line_width))
        self.color.setAlpha(self.nohover_alpha)
        self.setBrush(self.color)
        if layer is not None:
            self.setZValue(layer)
            for vertex in self.vertices:
                vertex.setZValue(layer)
        for vertex in self.vertices:
            vertex.setColor(color)

        self.setFlag(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable,
            not self.scene().mainwindow.annos_dock_widget.checkBox_lock.isChecked(),
        )

    def calculate_area(self) -> float:
        """Calculate area of polygon using the shoelace formula."""
        area = 0
        num_points = len(self.points)
        for i in range(num_points):
            p1 = self.points[i]
            p2 = self.points[(i + 1) % num_points]
            d = p1.x() * p2.y() - p2.x() * p1.y()
            area += d
        return abs(area) / 2

    # ---- serialisation -------------------------------------------------------

    def load_object(self, obj):
        """
        Load attributes from an Annotation Object.
        """
        segmentation = obj.segmentation
        for x, y in segmentation:
            point = QtCore.QPointF(x, y)
            self.addPoint(point)
        color = self.scene().mainwindow.category_color_dict.get(obj.category, "#6F737A")
        self.set_drawed(
            obj.category,
            obj.group,
            obj.iscrowd,
            obj.note,
            QtGui.QColor(color),
            obj.layer,
        )
        self.area = obj.area
        self.object_id = getattr(obj, "object_id", "")
        self.track_id = getattr(obj, "track_id", "")
        self.source = getattr(obj, "source", "manual")
        self.review_status = getattr(obj, "review_status", "reviewed")
        self.origin_source = getattr(obj, "origin_source", "")

    def to_object(self) -> Object:
        """Convert to an Annotation Object for serialisation."""
        if self.is_drawing:
            return None
        segmentation = []
        for point in self.points:
            point = point + self.pos()
            segmentation.append((round(point.x(), 2), round(point.y(), 2)))
        xmin = self.boundingRect().x() + self.pos().x()
        ymin = self.boundingRect().y() + self.pos().y()
        xmax = xmin + self.boundingRect().width()
        ymax = ymin + self.boundingRect().height()

        if (
            not self.scene().mainwindow.cfg["software"]["real_time_area"]
            or self.area == 0
        ):
            self.area = self.calculate_area()

        object = Object(
            self.category,
            group=self.group,
            segmentation=segmentation,
            area=self.area,
            layer=self.zValue(),
            bbox=(xmin, ymin, xmax, ymax),
            iscrowd=self.iscrowd,
            note=self.note,
            object_id=self.object_id,
            track_id=self.track_id,
            source=self.source,
            review_status=self.review_status,
            origin_source=self.origin_source,
        )
        return object

# ============================================================
#  OBB — Oriented Bounding Box annotation shape
# ============================================================

class OBB(QtWidgets.QGraphicsPolygonItem, BaseShape):
    """Oriented Bounding Box annotation.

    An OBB is a rectangle defined by 4 corner points.  Unlike a free-form
    Polygon, the 4 points are **constrained** to form a rectangle — dragging
    one corner keeps the opposite corner fixed and preserves the current
    orientation (rotation angle).

    Creation flow (3 clicks)::

        P0, P1  → define the first edge (direction + length).
        P_click → defines the perpendicular width; the click is projected
                   onto the perpendicular so that P0-P1-P3-P2 is a true
                   rectangle.

    Internal representation
    -----------------------
    ``self.points`` holds the 4 corner points in **local** coordinates in
    clockwise order ``[P0, P1, P3, P2]`` where::

        P0 → P1   first edge
        P1 → P3   perpendicular edge (width direction)
        P3 → P2   opposite edge   (parallel to P0→P1)
        P2 → P0   closing edge    (parallel to P1→P3)

    The angle / centre / size properties are **derived** from ``self.points``
    rather than stored — this keeps the geometry consistent with the points.
    """

    def __init__(self):
        QtWidgets.QGraphicsPolygonItem.__init__(self, parent=None)
        self._init_shape(OBBVertex)

        self.line_width = 1
        self.hover_alpha = 150
        self.nohover_alpha = 80
        self.category = ""
        self.group = 0
        self.iscrowd = False
        self.note = ""
        self.object_id = ""
        self.track_id = ""
        self.source = "manual"
        self.review_status = "reviewed"
        self.area = 0

        self.color = QtGui.QColor("#ff0000")
        self.is_drawing = True
        pen = QtGui.QPen(self.color, self.line_width)
        pen.setStyle(QtCore.Qt.PenStyle.DotLine)
        self.setPen(pen)
        self.setBrush(QtGui.QBrush(self.color, QtCore.Qt.BrushStyle.FDiagPattern))

        self.setAcceptHoverEvents(True)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True
        )
        self.setZValue(1e5)

    # ------------------------------------------------------------------
    #  Derived geometric properties
    # ------------------------------------------------------------------

    @property
    def angle(self) -> float:
        """Rotation angle of the first edge in radians (range [-π, π])."""
        if len(self.points) < 2:
            return 0.0
        d = self.points[1] - self.points[0]
        return math.atan2(d.y(), d.x())

    @property
    def center(self) -> QtCore.QPointF:
        """Centre point of the rectangle (local coordinates)."""
        if len(self.points) < 4:
            return QtCore.QPointF(0, 0)
        return (self.points[0] + self.points[2]) / 2

    @property
    def size(self):
        """Return ``(width, height)`` tuple.

        *width* — length of the first edge ``|P0→P1|``.
        *height* — length of the perpendicular edge ``|P0→P2|``.
        """
        if len(self.points) < 4:
            return (0, 0)
        w = math.hypot(
            self.points[1].x() - self.points[0].x(),
            self.points[1].y() - self.points[0].y(),
        )
        h = math.hypot(
            self.points[3].x() - self.points[0].x(),
            self.points[3].y() - self.points[0].y(),
        )
        return (w, h)

    # ------------------------------------------------------------------
    #  Point / vertex CRUD (override BaseShape)
    # ------------------------------------------------------------------

    def addPoint(self, point: QtCore.QPointF):
        """Append a corner point.

        Interactive drawing (3 clicks)::

            click 1 → [P0, T]          (anchor + trailing)
            click 2 → [P0, P1, T]      (2 anchors + trailing)
            click 3 → [P0, P1, P2]     → auto-complete to 4 corners

        During loading from disk the guard allows up to 4 points so that
        existing 4-point annotations are restored without modification.

        .. note::
            Trailing points (the live mouse-following point) should be
            added via :meth:`_add_trailing` instead so that they do **not**
            trigger auto-complete.
        """
        _loading = getattr(self, "_loading", False)
        max_points = 4 if _loading else 3
        if len(self.points) >= max_points:
            return

        super().addPoint(point)

        if _loading:
            return

        if len(self.points) == 3:
            self._complete_rectangle()
            self.redraw()
            self.is_drawing = False
            self.area = self.calculate_area()

    def _add_trailing(self, point: QtCore.QPointF):
        """Add a mouse-following trailing point **without** triggering auto-complete.

        This bypasses :meth:`addPoint` and calls :meth:`BaseShape.addPoint`
        directly so that the trailing point is a plain vertex that can be
        freely moved / removed without side-effects.
        """
        # pylint: disable=protected-access
        if len(self.points) >= 3:
            return
        super(OBB, self).addPoint(point)

    def _complete_rectangle(self):
        """Compute the true rectangle from 3 real corners.

        Called when ``self.points == [P0, P1, P_click]`` where:
        * P0, P1  — first edge (clicks 1 & 2)
        * P_click — 3rd corner (click 3), projected onto perpendicular through P1

        After this call ``self.points`` is ``[P0, P1, P3, P2]`` (clockwise).
        """
        p0 = self.points[0]
        p1 = self.points[1]
        p_click = self.points[2]  # 3rd corner, needs projection

        edge = p1 - p0
        # perpendicular: rotate edge by +90°
        d_perp = QtCore.QPointF(-edge.y(), edge.x())

        # Project p_click onto the perpendicular line from P1
        v = p_click - p1
        denom = d_perp.x() * d_perp.x() + d_perp.y() * d_perp.y()
        t = (v.x() * d_perp.x() + v.y() * d_perp.y()) / denom

        p3 = QtCore.QPointF(p1.x() + t * d_perp.x(), p1.y() + t * d_perp.y())
        p2 = p3 - edge  # == P0 + t * d_perp

        # Replace the clicked corner with its projected position (P3)
        self.points[2] = p3
        self.vertices[2].setPos(p3)

        # Append the 4th corner (P2)
        self.points.append(p2)
        vertex_size = self.scene().mainwindow.cfg["software"]["vertex_size"] * 2
        vertex = self._vertex_cls(self, self.color, vertex_size)
        self.scene().addItem(vertex)
        self.vertices.append(vertex)
        vertex.setPos(p2)

    def movePoint(self, index: int, point: QtCore.QPointF):
        """Move a corner while maintaining the rectangular constraint.

        The **opposite** corner ``(index+2)%4`` stays fixed.  The other two
        corners are recalculated so that the rectangle keeps its current
        orientation (``self.angle``).
        """
        if not 0 <= index < len(self.points):
            return
        if len(self.points) < 4:
            super().movePoint(index, point)
            return

        new_corner = self.mapFromScene(point)
        opposite_idx = (index + 2) % 4
        fixed_corner = self.points[opposite_idx]

        self._recompute_from_diagonal(index, new_corner, opposite_idx, fixed_corner)
        self.redraw()
        self._on_point_moved(index, point)

    def _recompute_from_diagonal(self, dragged_idx, new_pos, fixed_idx, fixed_pos):
        """Recompute all 4 corners from a new diagonal, preserving angle."""
        ang = self.angle
        d1 = QtCore.QPointF(math.cos(ang), math.sin(ang))   # edge direction
        d2 = QtCore.QPointF(-math.sin(ang), math.cos(ang))   # perpendicular

        center = (new_pos + fixed_pos) / 2
        half_diag = new_pos - center  # = (new_pos - fixed_pos) / 2

        hw = half_diag.x() * d1.x() + half_diag.y() * d1.y()  # dot(d1, half_diag)
        hh = half_diag.x() * d2.x() + half_diag.y() * d2.y()  # dot(d2, half_diag)

        self.points[dragged_idx] = new_pos
        self.points[fixed_idx] = fixed_pos
        self.points[(dragged_idx + 1) % 4] = QtCore.QPointF(
            center.x() - d1.x() * hw + d2.x() * hh,
            center.y() - d1.y() * hw + d2.y() * hh,
        )
        self.points[(dragged_idx + 3) % 4] = QtCore.QPointF(
            center.x() + d1.x() * hw - d2.x() * hh,
            center.y() + d1.y() * hw - d2.y() * hh,
        )

        # Sync vertex scene positions
        for i in range(4):
            if i != dragged_idx:
                self.moveVertex(i, self.mapToScene(self.points[i]))

    def rotate(self, delta_angle: float):
        """Rotate the OBB by *delta_angle* radians around its centre."""
        if len(self.points) < 4:
            return

        c = self.center
        cos_a = math.cos(delta_angle)
        sin_a = math.sin(delta_angle)

        for i in range(4):
            dx = self.points[i].x() - c.x()
            dy = self.points[i].y() - c.y()
            self.points[i] = QtCore.QPointF(
                c.x() + dx * cos_a - dy * sin_a,
                c.y() + dx * sin_a + dy * cos_a,
            )
            self.moveVertex(i, self.mapToScene(self.points[i]))

        self.redraw()

    def moveVertex(self, index, point):
        """Direct vertex position update (bypasses ``movePoint``)."""
        if not 0 <= index < len(self.vertices):
            return
        vertex = self.vertices[index]
        vertex.setEnabled(False)
        vertex.setPos(point)
        vertex.setEnabled(True)

    # ------------------------------------------------------------------
    #  Hook — side-effects after vertex drag
    # ------------------------------------------------------------------

    def _on_point_moved(self, index: int, point: QtCore.QPointF):
        if self.scene().mainwindow.cfg["software"]["real_time_area"]:
            self.area = self.calculate_area()
        if (
            self.scene().mainwindow.load_finished
            and not self.is_drawing
            and self.scene().mode != STATUSMode.REPAINT
        ):
            self.mark_edited()

    # ------------------------------------------------------------------
    #  Qt item events
    # ------------------------------------------------------------------

    def itemChange(
        self, change: "QGraphicsItem.GraphicsItemChange", value: typing.Any
    ):
        if (
            change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged
            and not self.is_drawing
            and self.scene().mode != STATUSMode.CREATE
        ):
            if self.isSelected():
                color = QtGui.QColor("#00A0FF")
                color.setAlpha(self.hover_alpha)
                self.setBrush(color)
                self.scene().selected_polygons_list.append(self)
            else:
                self.color.setAlpha(self.nohover_alpha)
                self.setBrush(self.color)
                if self in self.scene().selected_polygons_list:
                    self.scene().selected_polygons_list.remove(self)
            self.scene().mainwindow.annos_dock_widget.set_selected(self)

        if (
            change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionChange
        ):
            if self.is_drawing:
                value = QtCore.QPointF(0, 0)
            else:
                bias = value
                l, t, b, r = (
                    self.boundingRect().left(),
                    self.boundingRect().top(),
                    self.boundingRect().bottom(),
                    self.boundingRect().right(),
                )
                if l + bias.x() < 0:
                    bias.setX(-l)
                if r + bias.x() > self.scene().width():
                    bias.setX(self.scene().width() - r)
                if t + bias.y() < 0:
                    bias.setY(-t)
                if b + bias.y() > self.scene().height():
                    bias.setY(self.scene().height() - b)

                for index, point in enumerate(self.points):
                    self.moveVertex(index, point + bias)

                if self.scene().mainwindow.load_finished:
                    self.mark_edited()

        if (
            change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged
            and self.isSelected()
        ):
            self.setSelected(not self.is_drawing)

        return super().itemChange(change, value)

    def hoverEnterEvent(self, event: "QGraphicsSceneHoverEvent"):
        if not self.is_drawing and not self.isSelected():
            self.color.setAlpha(self.hover_alpha)
            self.setBrush(self.color)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event: "QGraphicsSceneHoverEvent"):
        if not self.is_drawing and not self.isSelected():
            self.color.setAlpha(self.nohover_alpha)
            self.setBrush(self.color)
        super().hoverLeaveEvent(event)

    def mouseDoubleClickEvent(self, event: "QGraphicsSceneMouseEvent"):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            editor = self.scene().mainwindow._ensure_category_edit_dialog()
            editor.polygons = [self]
            editor.load_cfg()
            editor.show()

    # ------------------------------------------------------------------
    #  Rendering
    # ------------------------------------------------------------------

    def redraw(self):
        if len(self.points) < 1:
            return
        self.setPolygon(QtGui.QPolygonF(self.points))

    def change_color(self, color: QtGui.QColor):
        self.color = color
        if not self.scene().mainwindow.cfg["software"]["show_edge"]:
            color.setAlpha(0)
        self.setPen(QtGui.QPen(color, self.line_width))
        self.color.setAlpha(self.nohover_alpha)
        self.setBrush(self.color)

        vertex_color = QtGui.QColor(self.color)
        vertex_color.setAlpha(255)
        for vertex in self.vertices:
            vertex.setPen(QtGui.QPen(vertex_color, self.line_width))
            vertex.setBrush(vertex_color)

    # ------------------------------------------------------------------
    #  Lifecycle
    # ------------------------------------------------------------------

    def set_drawed(
        self,
        category: str,
        group: int,
        iscrowd: bool,
        note: str,
        color: QtGui.QColor,
        layer: int = None,
    ):
        self.is_drawing = False
        self.category = category
        if isinstance(group, str):
            group = 0 if group == "" else int(group)
        self.group = group
        self.iscrowd = iscrowd
        self.note = note

        self.color = QtGui.QColor(color)
        self.color.setAlpha(255)

        if not self.scene().mainwindow.cfg["software"]["show_edge"]:
            self.color.setAlpha(0)
        self.setPen(QtGui.QPen(self.color, self.line_width))
        self.color.setAlpha(self.nohover_alpha)
        self.setBrush(self.color)
        if layer is not None:
            self.setZValue(layer)
            for vertex in self.vertices:
                vertex.setZValue(layer)
        for vertex in self.vertices:
            vertex.setColor(color)

        self.setFlag(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable,
            not self.scene().mainwindow.annos_dock_widget.checkBox_lock.isChecked(),
        )

    def calculate_area(self) -> float:
        """Return width × height."""
        w, h = self.size
        return w * h

    # ------------------------------------------------------------------
    #  Serialisation
    # ------------------------------------------------------------------

    def load_object(self, obj):
        """Load attributes from an Annotation Object."""
        self._loading = True
        for x, y in obj.segmentation:
            self.addPoint(QtCore.QPointF(x, y))
        self._loading = False

        if len(self.points) == 3:
            self._complete_rectangle()

        color = self.scene().mainwindow.category_color_dict.get(
            obj.category, "#6F737A"
        )
        self.set_drawed(
            obj.category,
            obj.group,
            obj.iscrowd,
            obj.note,
            QtGui.QColor(color),
            obj.layer,
        )
        self.area = obj.area
        self.object_id = getattr(obj, "object_id", "")
        self.track_id = getattr(obj, "track_id", "")
        self.source = getattr(obj, "source", "manual")
        self.review_status = getattr(obj, "review_status", "reviewed")
        self.origin_source = getattr(obj, "origin_source", "")

    def to_object(self) -> Object:
        """Convert to an Annotation Object for serialisation."""
        if self.is_drawing:
            return None

        segmentation = []
        for point in self.points:
            pt = point + self.pos()
            segmentation.append((round(pt.x(), 2), round(pt.y(), 2)))

        xmin = self.boundingRect().x() + self.pos().x()
        ymin = self.boundingRect().y() + self.pos().y()
        xmax = xmin + self.boundingRect().width()
        ymax = ymin + self.boundingRect().height()

        if (
            not self.scene().mainwindow.cfg["software"]["real_time_area"]
            or self.area == 0
        ):
            self.area = self.calculate_area()

        obj = Object(
            self.category,
            group=self.group,
            segmentation=segmentation,
            area=self.area,
            layer=self.zValue(),
            bbox=(xmin, ymin, xmax, ymax),
            iscrowd=self.iscrowd,
            note=self.note,
            is_obb=True,
            object_id=self.object_id,
            track_id=self.track_id,
            source=self.source,
            review_status=self.review_status,
            origin_source=self.origin_source,
        )
        return obj


# ============================================================
#  BBox — axis-aligned bounding box annotation shape
# ============================================================

class BBox(OBB):
    """Axis-aligned bounding box annotation.

    Drawn with two clicks or one drag between opposite corners.  The corners
    are stored like an OBB (``[TL, TR, BR, BL]``) with a zero edge angle, so
    dragging any corner keeps the opposite corner fixed and the box upright.
    Rotation is not offered; the shape serialises as ``shape_type="rectangle"``.
    """

    def addPoint(self, point: QtCore.QPointF):
        if getattr(self, "_loading", False):
            if len(self.points) < 4:
                BaseShape.addPoint(self, point)
            return
        if len(self.points) >= 2:
            return
        BaseShape.addPoint(self, point)

    def complete_from_corners(self, first: QtCore.QPointF, second: QtCore.QPointF) -> bool:
        """Replace the interactive anchor/trailing points with the final box."""

        left, right = sorted((first.x(), second.x()))
        top, bottom = sorted((first.y(), second.y()))
        if right - left < 1 or bottom - top < 1:
            return False
        self.delete()
        self._loading = True
        try:
            for x, y in rectangle_segmentation(left, top, right, bottom):
                self.addPoint(QtCore.QPointF(x, y))
        finally:
            self._loading = False
        self.redraw()
        self.is_drawing = False
        self.area = self.calculate_area()
        return True

    def redraw(self):
        if len(self.points) == 2:
            # Live preview between the anchor and the trailing corner.
            first, second = self.points
            self.setPolygon(QtGui.QPolygonF(QtCore.QRectF(first, second).normalized()))
            return
        super().redraw()

    def rotate(self, delta_angle: float):
        """Axis-aligned boxes never rotate."""

    def _bounds(self):
        xs = [point.x() + self.pos().x() for point in self.points]
        ys = [point.y() + self.pos().y() for point in self.points]
        return min(xs), min(ys), max(xs), max(ys)

    def calculate_area(self) -> float:
        if len(self.points) < 4:
            return 0
        xmin, ymin, xmax, ymax = self._bounds()
        return (xmax - xmin) * (ymax - ymin)

    def load_object(self, obj):
        xs = [x for x, _ in obj.segmentation]
        ys = [y for _, y in obj.segmentation]
        rectangle = rectangle_segmentation(min(xs), min(ys), max(xs), max(ys))
        self._loading = True
        try:
            for x, y in rectangle:
                self.addPoint(QtCore.QPointF(x, y))
        finally:
            self._loading = False
        color = self.scene().mainwindow.category_color_dict.get(obj.category, "#6F737A")
        self.set_drawed(obj.category, obj.group, obj.iscrowd, obj.note, QtGui.QColor(color), obj.layer)
        self.area = obj.area or self.calculate_area()
        self.object_id = getattr(obj, "object_id", "")
        self.track_id = getattr(obj, "track_id", "")
        self.source = getattr(obj, "source", "manual")
        self.review_status = getattr(obj, "review_status", "reviewed")
        self.origin_source = getattr(obj, "origin_source", "")

    def to_object(self) -> Object:
        if self.is_drawing or len(self.points) < 4:
            return None
        xmin, ymin, xmax, ymax = (round(value, 2) for value in self._bounds())
        self.area = self.calculate_area()
        return Object(
            self.category,
            group=self.group,
            segmentation=rectangle_segmentation(xmin, ymin, xmax, ymax),
            area=self.area,
            layer=self.zValue(),
            bbox=(xmin, ymin, xmax, ymax),
            iscrowd=self.iscrowd,
            note=self.note,
            object_id=self.object_id,
            track_id=self.track_id,
            source=self.source,
            review_status=self.review_status,
            origin_source=self.origin_source,
            shape_type="rectangle",
        )


def shape_for_object(obj):
    """Return an empty canvas shape matching a stored object's geometry type."""

    shape_type = getattr(obj, "shape_type", "obb" if getattr(obj, "is_obb", False) else "polygon")
    if shape_type == "rectangle":
        return BBox()
    if shape_type == "obb":
        return OBB()
    return Polygon()

# ============================================================
#  Line — repaint-mode guide line
# ============================================================

class Line(QtWidgets.QGraphicsPathItem, BaseShape):
    """Visual guide line shown during repaint mode."""

    def __init__(self):
        QtWidgets.QGraphicsPathItem.__init__(self, parent=None)
        self._init_shape(LineVertex)

        self.line_width = 1
        self.color = QtGui.QColor("#ff0000")
        pen = QtGui.QPen(self.color, self.line_width)
        pen.setStyle(QtCore.Qt.PenStyle.DotLine)
        self.setPen(pen)
        self.setZValue(1e5)

    def redraw(self):
        if len(self.points) < 1:
            return

        line_path = QtGui.QPainterPath()
        if self.points:
            line_path.moveTo(self.points[0])
            for point in self.points[1:]:
                line_path.lineTo(point)

        self.setPath(line_path)


# ============================================================
#  PromptRect — SAM box-prompt rectangle
# ============================================================

class PromptRect(QtWidgets.QGraphicsRectItem, BaseShape):
    """Visual rectangle for SAM box-prompt mode."""

    def __init__(self):
        QtWidgets.QGraphicsRectItem.__init__(self, parent=None)
        self._init_shape(PromptRectVertex)

        self.line_width = 1
        self.color = QtGui.QColor("#ff0000")

        pen = QtGui.QPen(self.color, self.line_width)
        pen.setStyle(QtCore.Qt.PenStyle.DotLine)
        self.setPen(pen)

    def redraw(self):
        if len(self.points) < 2:
            return

        self.setRect(QtCore.QRectF(self.points[0], self.points[-1]))
