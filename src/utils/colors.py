"""Catppuccin Mocha color palette and auto-assignment."""

# 20 distinct colors from Catppuccin Mocha
CATPPUCCIN_PALETTE = [
    "#a6e3a1",  # green
    "#89b4fa",  # blue
    "#f38ba8",  # red
    "#fab387",  # peach
    "#cba6f7",  # mauve
    "#f9e2af",  # yellow
    "#94e2d5",  # teal
    "#f5c2e7",  # pink
    "#89dceb",  # sky
    "#eba0ac",  # maroon
    "#74c7ec",  # sapphire
    "#b4befe",  # lavender
    "#a6adc8",  # subtext0
    "#f2cdcd",  # flamingo
    "#e6c384",  # gold (custom)
    "#c6a0f6",  # violet (custom)
    "#8caaee",  # blue2 (custom)
    "#e78284",  # red2 (custom)
    "#a5adce",  # overlay (custom)
    "#81c8be",  # teal2 (custom)
]


def assign_color(index: int) -> str:
    """Assign a color from the palette by index, wrapping around."""
    return CATPPUCCIN_PALETTE[index % len(CATPPUCCIN_PALETTE)]
