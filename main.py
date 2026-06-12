"""AutoLabel Dock — entry point."""
import sys

from PyQt5.QtWidgets import QApplication

from src.app import MainWindow
from src.ui.theme import apply_theme
from src.utils.logging_config import setup_logging


def main() -> int:
    setup_logging()
    app = QApplication(sys.argv)
    apply_theme(app)
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
