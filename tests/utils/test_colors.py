"""Tests for Catppuccin color palette utility."""
from src.utils.colors import CATPPUCCIN_PALETTE, assign_color


class TestCatppuccinPalette:
    def test_palette_has_20_colors(self):
        assert len(CATPPUCCIN_PALETTE) == 20

    def test_colors_are_hex(self):
        for color in CATPPUCCIN_PALETTE:
            assert color.startswith("#")
            assert len(color) == 7

    def test_assign_color_by_index(self):
        assert assign_color(0) == CATPPUCCIN_PALETTE[0]
        assert assign_color(5) == CATPPUCCIN_PALETTE[5]

    def test_assign_color_wraps_around(self):
        assert assign_color(20) == CATPPUCCIN_PALETTE[0]
        assert assign_color(25) == CATPPUCCIN_PALETTE[5]
