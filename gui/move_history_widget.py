from PyQt6.QtWidgets import QListWidget, QListWidgetItem
from PyQt6.QtCore import Qt


class MoveHistoryWidget(QListWidget):
    """Widget to display move history"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        """Configure the widget"""
        self.setStyleSheet("""
            QListWidget {
                background-color: #2A2A2A;
                color: white;
                border: 1px solid #666666;
                font-family: 'Courier New', monospace;
                font-size: 12px;
            }
            QListWidget::item {
                padding: 5px;
            }
            QListWidget::item:selected {
                background-color: #4A4A4A;
            }
        """)
        
    def add_move(self, move_description):
        """Add a move to the history"""
        item = QListWidgetItem(move_description)
        self.addItem(item)
        # Auto-scroll to bottom
        self.scrollToBottom()
        
    def clear_history(self):
        """Clear all moves"""
        self.clear()

