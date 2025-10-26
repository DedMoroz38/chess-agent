#!/usr/bin/env python3
"""
Chess Fragment GUI - Entry Point
Launches the PyQt6 GUI for the 5x5 Chess Fragment game
"""
import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from gui.main_window import ChessMainWindow


def main():
    """Launch the Chess Fragment GUI"""
    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Chess Fragment")
    app.setOrganizationName("COMP2321")
    
    # Create and show main window
    window = ChessMainWindow()
    window.show()
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

