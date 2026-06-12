"""Tests for AnnotationPanel."""
import pytest
from PyQt5.QtCore import Qt


class TestAnnotationPanel:
    def test_set_annotations(self, qapp):
        from src.ui.properties import AnnotationPanel
        from src.core.annotation import Annotation

        panel = AnnotationPanel()
        anns = [
            Annotation(class_name="cat", class_id=0, bbox=(0.5, 0.5, 0.3, 0.4)),
            Annotation(class_name="dog", class_id=1, bbox=(0.2, 0.3, 0.1, 0.2)),
        ]
        panel.set_annotations(anns)
        assert panel._ann_tree.topLevelItemCount() == 2

    def test_set_annotations_with_keypoints(self, qapp):
        from src.ui.properties import AnnotationPanel
        from src.core.annotation import Annotation, Keypoint

        panel = AnnotationPanel()
        kps = [Keypoint(x=0.1, y=0.2, visible=2, label="nose"),
               Keypoint(x=0.3, y=0.4, visible=1, label="eye")]
        ann = Annotation(class_name="person", class_id=0, bbox=(0.5, 0.5, 0.3, 0.4), keypoints=kps)
        panel.set_annotations([ann])
        assert panel._ann_tree.topLevelItemCount() == 1
        top = panel._ann_tree.topLevelItem(0)
        assert top.childCount() == 2
        assert top.isExpanded()

    def test_clear(self, qapp):
        from src.ui.properties import AnnotationPanel
        from src.core.annotation import Annotation

        panel = AnnotationPanel()
        ann = Annotation(class_name="cat", class_id=0, bbox=(0.5, 0.5, 0.3, 0.4))
        panel.set_annotations([ann])
        panel.clear()
        assert panel._ann_tree.topLevelItemCount() == 0

    def test_select_annotation_shows_properties(self, qapp):
        from src.ui.properties import AnnotationPanel
        from src.core.annotation import Annotation

        panel = AnnotationPanel()
        ann = Annotation(class_name="cat", class_id=0, bbox=(0.5, 0.5, 0.3, 0.4), confidence=0.95)
        panel.set_annotations([ann])
        panel.select_annotation(ann.id)
        assert "cat" in panel._class_label.text()

    def test_select_none_clears_properties(self, qapp):
        from src.ui.properties import AnnotationPanel
        from src.core.annotation import Annotation

        panel = AnnotationPanel()
        ann = Annotation(class_name="cat", class_id=0, bbox=(0.5, 0.5, 0.3, 0.4))
        panel.set_annotations([ann])
        panel.select_annotation(ann.id)
        panel.select_annotation(None)
        assert panel._class_label.text() == ""

    def test_select_keypoint_shows_properties(self, qapp):
        from src.ui.properties import AnnotationPanel
        from src.core.annotation import Annotation, Keypoint

        panel = AnnotationPanel()
        kp = Keypoint(x=0.1, y=0.2, visible=2, label="nose")
        ann = Annotation(class_name="person", class_id=0, bbox=(0.5, 0.5, 0.3, 0.4), keypoints=[kp])
        panel.set_annotations([ann])
        panel.select_keypoint(ann.id, 0)
        assert "nose" in panel._class_label.text()
        assert "person" in panel._conf_label.text()

    def test_set_image_tags_removed(self, qapp):
        """set_image_tags / get_image_tags were removed when the image-tags
        widget was repurposed into a project class list. This guards against
        accidental re-introduction."""
        from src.ui.properties import AnnotationPanel

        panel = AnnotationPanel()
        assert not hasattr(panel, "set_image_tags")
        assert not hasattr(panel, "get_image_tags")

    def test_project_class_list_rendered(self, qapp):
        from PyQt5.QtWidgets import QLabel
        from src.ui.properties import AnnotationPanel

        panel = AnnotationPanel()
        panel.set_class_colors({"cat": "#a6e3a1", "dog": "#f38ba8"})
        panel.set_classes(["cat", "dog"])

        assert panel._classes_list.count() == 2
        first = panel._classes_list.item(0)
        from PyQt5.QtCore import Qt as _Qt
        assert first.data(_Qt.UserRole) == "cat"
        row = panel._classes_list.itemWidget(first)
        name_lbl = row.findChild(QLabel, "name_lbl")
        count_lbl = row.findChild(QLabel, "count_lbl")
        assert name_lbl is not None and count_lbl is not None
        assert name_lbl.text().startswith("0")  # index 0 prefix
        assert "cat" in name_lbl.text()
        assert count_lbl.text() == "×0"

    def test_project_class_list_counts_update_with_annotations(self, qapp):
        from PyQt5.QtWidgets import QLabel
        from src.ui.properties import AnnotationPanel
        from src.core.annotation import Annotation

        panel = AnnotationPanel()
        panel.set_classes(["cat", "dog"])
        panel.set_annotations([
            Annotation(class_name="cat", class_id=0, bbox=(0.5, 0.5, 0.3, 0.4)),
            Annotation(class_name="cat", class_id=0, bbox=(0.1, 0.1, 0.2, 0.2)),
            Annotation(class_name="dog", class_id=1, bbox=(0.2, 0.3, 0.1, 0.2)),
        ])
        cat_row = panel._classes_list.itemWidget(panel._classes_list.item(0))
        dog_row = panel._classes_list.itemWidget(panel._classes_list.item(1))
        assert cat_row.findChild(QLabel, "count_lbl").text() == "×2"
        assert dog_row.findChild(QLabel, "count_lbl").text() == "×1"

    def test_keypoint_count_suffix_when_pose(self, qapp):
        """Pose annotations contribute keypoint totals; detect-only does not."""
        from PyQt5.QtWidgets import QLabel
        from src.ui.properties import AnnotationPanel
        from src.core.annotation import Annotation, Keypoint

        panel = AnnotationPanel()
        panel.set_classes(["person"])

        # No keypoints → no `(N kp)` suffix
        panel.set_annotations([
            Annotation(class_name="person", class_id=0, bbox=(0.5, 0.5, 0.2, 0.2)),
        ])
        row = panel._classes_list.itemWidget(panel._classes_list.item(0))
        assert row.findChild(QLabel, "count_lbl").text() == "×1"

        # 2 annotations × 3 keypoints each → "×2  (6 kp)"
        kps = [
            Keypoint(x=0.1, y=0.2, visible=2, label="nose"),
            Keypoint(x=0.2, y=0.3, visible=2, label="eye"),
            Keypoint(x=0.3, y=0.4, visible=1, label="ear"),
        ]
        panel.set_annotations([
            Annotation(class_name="person", class_id=0, bbox=(0.5, 0.5, 0.2, 0.2), keypoints=list(kps)),
            Annotation(class_name="person", class_id=0, bbox=(0.1, 0.1, 0.2, 0.2), keypoints=list(kps)),
        ])
        text = row.findChild(QLabel, "count_lbl").text()
        assert "×2" in text
        assert "6 kp" in text

    def test_default_class_toggle_signal_and_highlight(self, qapp):
        from PyQt5.QtWidgets import QLabel
        from src.ui.properties import AnnotationPanel

        panel = AnnotationPanel()
        panel.set_classes(["cat", "dog"])

        received: list = []
        panel.default_class_changed.connect(lambda name: received.append(name))

        def name_lbl(i):
            return panel._classes_list.itemWidget(
                panel._classes_list.item(i)
            ).findChild(QLabel, "name_lbl")

        # First double-click → set "dog"
        panel._on_class_double_clicked("dog")
        assert received == ["dog"]
        assert name_lbl(1).font().bold()
        assert not name_lbl(0).font().bold()

        # Second double-click on the SAME class → toggle off (emit None)
        panel._on_class_double_clicked("dog")
        assert received == ["dog", None]
        assert not name_lbl(1).font().bold()
        assert not name_lbl(0).font().bold()

        # API setter still works without re-firing
        panel.set_default_class("cat")
        assert name_lbl(0).font().bold()
        assert not name_lbl(1).font().bold()
        assert received == ["dog", None]

    def test_row_widget_emits_double_click_signal(self, qapp):
        """Verify the per-row widget actually wires its double-click into the
        panel's default-class flow (not relying on QListWidget.itemDoubleClicked)."""
        from PyQt5.QtCore import QEvent, Qt as _Qt, QPoint
        from PyQt5.QtGui import QMouseEvent
        from PyQt5.QtWidgets import QApplication
        from src.ui.properties import AnnotationPanel

        panel = AnnotationPanel()
        panel.set_classes(["cat", "dog"])

        received: list = []
        panel.default_class_changed.connect(lambda name: received.append(name))

        row = panel._classes_list.itemWidget(panel._classes_list.item(1))
        # Synthesize a real mouseDoubleClickEvent on the row widget
        ev = QMouseEvent(
            QEvent.MouseButtonDblClick,
            QPoint(5, 5),
            _Qt.LeftButton,
            _Qt.LeftButton,
            _Qt.NoModifier,
        )
        QApplication.sendEvent(row, ev)
        assert received == ["dog"]

    def test_stats_display(self, qapp):
        from src.ui.properties import AnnotationPanel
        from src.core.annotation import Annotation

        panel = AnnotationPanel()
        anns = [
            Annotation(class_name="cat", class_id=0, bbox=(0.5, 0.5, 0.3, 0.4), confirmed=True),
            Annotation(class_name="dog", class_id=1, bbox=(0.2, 0.3, 0.1, 0.2), confirmed=False),
            Annotation(class_name="cat", class_id=0, bbox=(0.7, 0.7, 0.2, 0.2), confirmed=True),
        ]
        panel.set_annotations(anns)
        assert "2" in panel._stats_label.text()  # 2 confirmed


class TestProjectStats:
    def test_set_project_stats_displays_totals(self, qapp):
        from src.ui.properties import AnnotationPanel

        panel = AnnotationPanel()
        stats = {
            "total_images": 100,
            "labeled_images": 80,
            "confirmed_images": 50,
            "total_annotations": 200,
            "class_counts": {"cat": 120, "dog": 80},
        }
        panel.set_project_stats(stats)
        assert "100" in panel._project_total_label.text()
        assert "80" in panel._project_labeled_label.text()
        assert "50" in panel._project_confirmed_label.text()
        assert "200" in panel._project_ann_count_label.text()
        assert panel._class_dist_list.count() == 2

    def test_set_project_stats_empty(self, qapp):
        from src.ui.properties import AnnotationPanel

        panel = AnnotationPanel()
        panel.set_project_stats({})
        assert "0" in panel._project_total_label.text()
        assert panel._class_dist_list.count() == 0

    def test_class_distribution_sorted_by_count(self, qapp):
        from src.ui.properties import AnnotationPanel

        panel = AnnotationPanel()
        stats = {
            "total_images": 10,
            "class_counts": {"rare": 5, "common": 50, "mid": 20},
        }
        panel.set_project_stats(stats)
        assert panel._class_dist_list.count() == 3
        # First item should be 'common' (highest count)
        assert "common" in panel._class_dist_list.item(0).text()
        assert "mid" in panel._class_dist_list.item(1).text()
        assert "rare" in panel._class_dist_list.item(2).text()

    def test_class_distribution_uses_colors(self, qapp):
        from src.ui.properties import AnnotationPanel

        panel = AnnotationPanel()
        panel.set_class_colors({"cat": "#a6e3a1"})
        stats = {"class_counts": {"cat": 10}}
        panel.set_project_stats(stats)
        item = panel._class_dist_list.item(0)
        assert item.foreground().color().name() == "#a6e3a1"


class TestAnnotationPanelLayout:
    _EXPECTED_TITLES = ("项目类别", "标注列表", "属性", "Tag", "项目统计")

    def test_splitter_holds_five_collapsible_panes(self, qapp):
        from PyQt5.QtWidgets import QSplitter
        from src.ui.collapsible_group import CollapsibleGroupBox
        from src.ui.properties import AnnotationPanel

        panel = AnnotationPanel()
        splitter = panel.findChild(QSplitter)
        assert splitter is not None
        assert splitter.count() == 5
        titles = [splitter.widget(i).title() for i in range(5)]
        assert tuple(titles) == self._EXPECTED_TITLES
        for i in range(5):
            assert isinstance(splitter.widget(i), CollapsibleGroupBox)

    def test_splitter_disallows_children_collapse_to_zero(self, qapp):
        from PyQt5.QtWidgets import QSplitter
        from src.ui.properties import AnnotationPanel

        panel = AnnotationPanel()
        splitter = panel.findChild(QSplitter)
        assert splitter.childrenCollapsible() is False

    def test_lists_have_higher_stretch_factor_than_aux(self, qapp):
        """标注列表 has the largest stretch factor (2) and should grow the
        most when the panel is given excess vertical space."""
        from PyQt5.QtWidgets import QSplitter
        from src.ui.properties import AnnotationPanel

        panel = AnnotationPanel()
        panel.resize(280, 1500)
        panel.show()
        qapp.processEvents()
        splitter = panel.findChild(QSplitter)
        sizes = splitter.sizes()
        # 标注列表 (factor 2) outgrows 项目类别 (factor 1).
        assert sizes[1] > sizes[0], f"sizes={sizes}"
        # And 标注列表 outgrows every aux pane.
        assert sizes[1] > sizes[2], f"sizes={sizes}"
        assert sizes[1] > sizes[3], f"sizes={sizes}"
        assert sizes[1] > sizes[4], f"sizes={sizes}"

    def test_stats_label_lives_inside_attribute_pane(self, qapp):
        from PyQt5.QtWidgets import QSplitter
        from src.ui.properties import AnnotationPanel

        panel = AnnotationPanel()
        splitter = panel.findChild(QSplitter)
        attr_pane = splitter.widget(2)  # 属性
        stats_label = panel._stats_label
        ancestor = stats_label.parent()
        while ancestor is not None and ancestor is not attr_pane:
            ancestor = ancestor.parent()
        assert ancestor is attr_pane

    def test_existing_set_annotations_still_works(self, qapp):
        from src.ui.properties import AnnotationPanel
        from src.core.annotation import Annotation

        panel = AnnotationPanel()
        anns = [
            Annotation(class_name="cat", class_id=0, bbox=(0.5, 0.5, 0.3, 0.4)),
            Annotation(class_name="dog", class_id=1, bbox=(0.2, 0.3, 0.1, 0.2)),
        ]
        panel.set_annotations(anns)
        assert panel._ann_tree.topLevelItemCount() == 2


class TestAnnotationPanelState:
    def test_save_state_default(self, qapp):
        from src.ui.properties import AnnotationPanel

        panel = AnnotationPanel()
        state = panel.save_state()
        assert "sizes" in state
        assert "collapsed" in state
        assert len(state["sizes"]) == 5
        assert state["collapsed"] == {
            "项目类别": False, "标注列表": False, "属性": False,
            "Tag": False, "项目统计": False,
        }

    def test_restore_state_applies_collapsed(self, qapp):
        from src.ui.properties import AnnotationPanel

        panel = AnnotationPanel()
        panel.restore_state({
            "sizes": [],
            "collapsed": {"属性": True, "项目统计": True},
        })
        assert panel._sections["属性"].isExpanded() is False
        assert panel._sections["项目统计"].isExpanded() is False
        assert panel._sections["标注列表"].isExpanded() is True

    def test_restore_state_applies_sizes_only_if_length_matches(self, qapp):
        from src.ui.properties import AnnotationPanel

        panel = AnnotationPanel()
        captured: list[list[int]] = []
        original = panel._splitter.setSizes

        def spy(sizes):
            captured.append(list(sizes))
            return original(sizes)

        panel._splitter.setSizes = spy

        # Correct length → forwarded.
        panel.restore_state({"sizes": [50, 60, 70, 80, 90], "collapsed": {}})
        assert captured == [[50, 60, 70, 80, 90]]

        # Wrong length → not forwarded.
        panel.restore_state({"sizes": [1, 2, 3], "collapsed": {}})
        assert captured == [[50, 60, 70, 80, 90]]

    def test_restore_state_round_trip(self, qapp):
        """save_state preserves whatever sizes QSplitter currently reports,
        and restore_state preserves the collapsed flags exactly."""
        from src.ui.properties import AnnotationPanel

        panel = AnnotationPanel()
        panel.restore_state({"collapsed": {"Tag": True}})
        snapshot = panel.save_state()
        # Sizes are whatever the live splitter reports (5 ints summing to its
        # current height); we don't pin exact pixel values here.
        assert isinstance(snapshot["sizes"], list)
        assert len(snapshot["sizes"]) == 5
        assert all(isinstance(s, int) for s in snapshot["sizes"])
        # Collapsed map round-trips faithfully.
        assert snapshot["collapsed"]["Tag"] is True
        for name in ("项目类别", "标注列表", "属性", "项目统计"):
            assert snapshot["collapsed"][name] is False

    def test_restore_state_ignores_unknown_section(self, qapp):
        from src.ui.properties import AnnotationPanel

        panel = AnnotationPanel()
        panel.restore_state({"sizes": [], "collapsed": {"未知区块": True}})
        for box in panel._sections.values():
            assert box.isExpanded() is True

    def test_restore_state_empty_dict_noop(self, qapp):
        from src.ui.properties import AnnotationPanel

        panel = AnnotationPanel()
        before_sizes = panel._splitter.sizes()
        panel.restore_state({})
        assert panel._splitter.sizes() == before_sizes
        for box in panel._sections.values():
            assert box.isExpanded() is True

    def test_collapse_shrinks_pane_and_grows_siblings(self, qapp):
        """Collapsing a pane must actually shrink its splitter slot — not
        just hide the body — and the freed space must go to siblings."""
        from src.ui.properties import AnnotationPanel

        panel = AnnotationPanel()
        panel.resize(280, 1200)
        panel.show()
        qapp.processEvents()

        before = panel._splitter.sizes()
        total_before = sum(before)

        # Collapse 项目统计 (idx 4).
        panel._sections["项目统计"].setExpanded(False)
        qapp.processEvents()
        after = panel._splitter.sizes()

        # The collapsed pane is now tiny (clamped to its header).
        assert after[4] < 40, f"项目统计 still occupies {after[4]}px after collapse"
        # The total stays the same (no space leaked).
        assert abs(sum(after) - total_before) <= 1
        # At least one expanded sibling grew.
        siblings_before = [before[i] for i in range(4)]
        siblings_after = [after[i] for i in range(4)]
        assert sum(siblings_after) > sum(siblings_before)

    def test_expand_then_collapse_then_expand_round_trips(self, qapp):
        """Expanding a previously-collapsed pane restores it to a useful
        size (not 0)."""
        from src.ui.properties import AnnotationPanel

        panel = AnnotationPanel()
        panel.resize(280, 1200)
        panel.show()
        qapp.processEvents()

        panel._sections["属性"].setExpanded(False)
        qapp.processEvents()
        panel._sections["属性"].setExpanded(True)
        qapp.processEvents()

        sizes = panel._splitter.sizes()
        assert sizes[2] > 30, f"属性 should be usable after re-expand, got {sizes[2]}"
