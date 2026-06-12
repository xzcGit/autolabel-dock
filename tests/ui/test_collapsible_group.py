"""Tests for CollapsibleGroupBox."""
from PyQt5.QtWidgets import QLabel, QVBoxLayout


class TestCollapsibleGroupBox:
    def test_default_expanded(self, qapp):
        from src.ui.collapsible_group import CollapsibleGroupBox

        box = CollapsibleGroupBox("Section")
        assert box.isExpanded() is True
        assert box.title() == "Section"

    def test_set_content_layout_parents_children(self, qapp):
        from src.ui.collapsible_group import CollapsibleGroupBox

        box = CollapsibleGroupBox("S")
        inner = QVBoxLayout()
        inner.addWidget(QLabel("hello"))
        box.set_content_layout(inner)
        labels = box.findChildren(QLabel)
        assert any(lbl.text() == "hello" for lbl in labels)

    def test_collapse_hides_body_emits_signal(self, qapp):
        from src.ui.collapsible_group import CollapsibleGroupBox

        box = CollapsibleGroupBox("S")
        inner = QVBoxLayout()
        inner.addWidget(QLabel("body"))
        box.set_content_layout(inner)
        box.show()
        qapp.processEvents()
        expanded_height = box.sizeHint().height()

        captured = []
        box.toggled.connect(captured.append)
        box.setExpanded(False)

        assert box.isExpanded() is False
        assert captured == [False]
        body_label = next(lbl for lbl in box.findChildren(QLabel) if lbl.text() == "body")
        assert body_label.isVisible() is False
        assert box.sizeHint().height() < expanded_height

    def test_expand_restores_body_emits_signal(self, qapp):
        from src.ui.collapsible_group import CollapsibleGroupBox

        box = CollapsibleGroupBox("S")
        inner = QVBoxLayout()
        inner.addWidget(QLabel("body"))
        box.set_content_layout(inner)
        box.show()
        qapp.processEvents()
        box.setExpanded(False)
        qapp.processEvents()
        captured = []
        box.toggled.connect(captured.append)

        box.setExpanded(True)
        qapp.processEvents()

        assert box.isExpanded() is True
        assert captured == [True]
        body_label = next(lbl for lbl in box.findChildren(QLabel) if lbl.text() == "body")
        assert body_label.isVisible() is True

    def test_set_expanded_idempotent_no_signal(self, qapp):
        from src.ui.collapsible_group import CollapsibleGroupBox

        box = CollapsibleGroupBox("S")
        captured = []
        box.toggled.connect(captured.append)
        box.setExpanded(True)  # already expanded
        assert captured == []

    def test_header_button_toggles_state(self, qapp):
        from PyQt5.QtWidgets import QToolButton
        from src.ui.collapsible_group import CollapsibleGroupBox

        box = CollapsibleGroupBox("S")
        btn = box.findChild(QToolButton)
        assert btn is not None
        btn.click()
        assert box.isExpanded() is False
        btn.click()
        assert box.isExpanded() is True

    def test_collapsed_clamps_max_height_to_header(self, qapp):
        """When collapsed, the widget's maximumHeight is clamped to the
        header height so a parent QSplitter releases the freed space."""
        from PyQt5.QtWidgets import QToolButton
        from src.ui.collapsible_group import CollapsibleGroupBox

        box = CollapsibleGroupBox("S")
        inner = QVBoxLayout()
        inner.addWidget(QLabel("body" * 20))  # tall body
        box.set_content_layout(inner)
        box.show()
        qapp.processEvents()

        expanded_max = box.maximumHeight()  # default ~16777215
        box.setExpanded(False)
        qapp.processEvents()

        collapsed_max = box.maximumHeight()
        header_h = box.findChild(QToolButton).sizeHint().height()
        assert collapsed_max <= header_h + 4
        assert collapsed_max < expanded_max

    def test_expanded_releases_max_height(self, qapp):
        """After expanding again the maximumHeight cap is released."""
        from src.ui.collapsible_group import CollapsibleGroupBox

        box = CollapsibleGroupBox("S")
        box.show()
        qapp.processEvents()
        original_max = box.maximumHeight()

        box.setExpanded(False)
        qapp.processEvents()
        assert box.maximumHeight() < original_max

        box.setExpanded(True)
        qapp.processEvents()
        assert box.maximumHeight() == original_max
