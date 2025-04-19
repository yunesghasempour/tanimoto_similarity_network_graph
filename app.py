import sys
import io
import numpy as np
import networkx as nx
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QTextEdit, QPushButton, QSplitter, 
                            QMessageBox, QSizePolicy, QFrame, QSlider, QGroupBox, QScrollArea)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit import DataStructs

# Color constants
YELLOW = "#ffd700"
BLACK = "#1a1a1a"
DARK_YELLOW = "#ccac00"
LIGHT_BLACK = "#333333"
WHITE = "#ffffff"

class MoleculePreviewWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {BLACK};
                color: {YELLOW};
                border: 2px solid {YELLOW};
                border-radius: 10px;
                padding: 10px;
            }}
            QLabel {{
                color: {YELLOW};
                font-weight: bold;
                font-size: 12pt;
            }}
            QGroupBox {{
                background-color: {LIGHT_BLACK};
                color: {YELLOW};
                border: 1px solid {YELLOW};
                border-radius: 5px;
                margin-top: 10px;
                padding: 5px;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)
        
        title = QLabel("Molecular Structure")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 14pt; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # First molecule group
        mol1_group = QGroupBox("First Molecule")
        mol1_layout = QVBoxLayout()
        self.mol1_label = QLabel()
        self.mol1_label.setAlignment(Qt.AlignCenter)
        self.mol1_label.setMinimumSize(250, 250)  # Set minimum size for square display
        self.mol1_smiles = QLabel()
        self.mol1_smiles.setAlignment(Qt.AlignCenter)
        self.mol1_smiles.setWordWrap(True)
        mol1_layout.addWidget(self.mol1_label)
        mol1_layout.addWidget(self.mol1_smiles)
        mol1_group.setLayout(mol1_layout)
        
        # Second molecule group
        mol2_group = QGroupBox("Second Molecule")
        mol2_layout = QVBoxLayout()
        self.mol2_label = QLabel()
        self.mol2_label.setAlignment(Qt.AlignCenter)
        self.mol2_label.setMinimumSize(250, 250)  # Set minimum size for square display
        self.mol2_smiles = QLabel()
        self.mol2_smiles.setAlignment(Qt.AlignCenter)
        self.mol2_smiles.setWordWrap(True)
        mol2_layout.addWidget(self.mol2_label)
        mol2_layout.addWidget(self.mol2_smiles)
        mol2_group.setLayout(mol2_layout)
        
        layout.addWidget(mol1_group)
        layout.addWidget(mol2_group)
        
        self.setFixedSize(400, 800)  # Adjust overall widget size
    
    def update_molecules(self, mol1, mol2, smiles1=None, smiles2=None, similarity=None):
        if mol1 and mol2:
            img1 = Draw.MolToImage(mol1, size=(250, 250))  # Make molecule images square
            img2 = Draw.MolToImage(mol2, size=(250, 250))  # Make molecule images square
            
            self.mol1_label.setPixmap(self.pil_to_pixmap(img1))
            self.mol2_label.setPixmap(self.pil_to_pixmap(img2))
            
            if smiles1:
                self.mol1_smiles.setText(f"SMILE: {smiles1}")
            if smiles2:
                self.mol2_smiles.setText(f"SMILE: {smiles2}")
            
            if similarity is not None:
                self.mol1_smiles.setText(f"Similarity: {similarity:.3f}\n{self.mol1_smiles.text()}")
        else:
            self.mol1_label.clear()
            self.mol2_label.clear()
            self.mol1_smiles.clear()
            self.mol2_smiles.clear()
    
    def pil_to_pixmap(self, pil_image):
        img_data = pil_image.convert("RGBA").tobytes("raw", "RGBA")
        qim = QImage(img_data, pil_image.size[0], pil_image.size[1], QImage.Format_RGBA8888)
        return QPixmap.fromImage(qim)

class SimpleNetworkCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(12, 10), dpi=100)
        self.fig.patch.set_facecolor(BLACK)
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor(BLACK)
        
        super(SimpleNetworkCanvas, self).__init__(self.fig)
        self.setParent(parent)
        
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(800, 600)
        self.updateGeometry()
        
        self.mpl_connect('button_press_event', self.on_click)
        
        self.molecules = None
        self.similarity_matrix = None
        self.smiles_list = None
        self.preview_widget = None
        self.threshold = 0.7
        self.node_positions = {}
        self.edge_list = []
        self.edge_lines = []  # Store the line objects for edge detection
    
    def on_click(self, event):
        if not hasattr(self, 'node_positions') or not self.node_positions:
            return
            
        if event.xdata is None or event.ydata is None:
            return
            
        click_pos = np.array([event.xdata, event.ydata])
        
        # Check for clicked edges
        for i, (start_node, end_node) in enumerate(self.edge_list):
            start_pos = self.node_positions[start_node]
            end_pos = self.node_positions[end_node]
            
            # Calculate distance from click to line segment
            line_vec = end_pos - start_pos
            click_vec = click_pos - start_pos
            line_len = np.sqrt(np.sum(line_vec ** 2))
            line_unit_vec = line_vec / line_len
            
            # Project click point onto line
            proj_len = np.dot(click_vec, line_unit_vec)
            proj_len = max(0, min(line_len, proj_len))  # Clamp to line segment
            proj_point = start_pos + line_unit_vec * proj_len
            
            # Check if click is close enough to line
            dist_to_line = np.sqrt(np.sum((click_pos - proj_point) ** 2))
            if dist_to_line < 0.1:  # Threshold for edge click detection
                if self.preview_widget:
                    sim = self.similarity_matrix[start_node, end_node]
                    self.preview_widget.update_molecules(
                        self.molecules[start_node],
                        self.molecules[end_node],
                        self.smiles_list[start_node],
                        self.smiles_list[end_node],
                        sim
                    )
                return
    
    def plot_network(self, similarity_matrix, molecules, smiles_list):
        self.molecules = molecules
        self.similarity_matrix = similarity_matrix
        self.smiles_list = smiles_list
        
        self.axes.clear()
        self.axes.set_facecolor(BLACK)
        
        n = len(molecules)
        
        # ساخت گراف با استفاده از NetworkX
        G = nx.Graph()
        
        # اضافه کردن گره‌ها
        for i in range(n):
            G.add_node(i)
        
        # اضافه کردن یال‌ها با وزن مبتنی بر شباهت
        self.edge_list = []
        for i in range(n):
            for j in range(i+1, n):
                if similarity_matrix[i, j] >= self.threshold:
                    G.add_edge(i, j, weight=similarity_matrix[i, j])
                    self.edge_list.append((i, j))
        
        # محاسبه موقعیت گره‌ها با استفاده از الگوریتم force-directed
        pos = nx.spring_layout(G, k=1.5, iterations=50, weight='weight')
        self.node_positions = pos
        
        # رسم یال‌ها
        self.edge_lines = []
        for i, j in self.edge_list:
            x1, y1 = pos[i]
            x2, y2 = pos[j]
            # رنگ‌آمیزی بر اساس میزان شباهت
            color = plt.cm.YlOrBr(similarity_matrix[i, j])
            line = self.axes.plot([x1, x2], [y1, y2], color=color, linewidth=2, alpha=0.7)[0]
            self.edge_lines.append(line)
        
        # رسم گره‌ها
        # اندازه گره‌ها بر اساس تعداد اتصالات
        node_sizes = [G.degree(node) * 200 + 500 for node in G.nodes()]
        
        for i in range(n):
            x, y = pos[i]
            size = node_sizes[i] / 100  # تبدیل اندازه به مقیاس مناسب برای markersize
            self.axes.plot(x, y, 'o', color=YELLOW, markersize=size)
            self.axes.text(x, y, f'M{i+1}', color='black', fontsize=12, 
                         fontweight='bold', ha='center', va='center')
        
        self.axes.set_title("Molecular Similarity Network", color=YELLOW, pad=20, fontsize=14)
        
        # اضافه کردن آمار
        stats = f"Molecules: {n}\nConnections: {len(self.edge_list)}"
        self.axes.text(0.02, 0.98, stats, transform=self.axes.transAxes,
                      color=YELLOW, fontsize=12, va='top')
        
        # تنظیم محدوده نمایش
        self.axes.set_xlim(-1.5, 1.5)
        self.axes.set_ylim(-1.5, 1.5)
        self.axes.axis('off')
        
        self.draw()
    
    def update_threshold(self, threshold):
        self.threshold = threshold
        if self.similarity_matrix is not None:
            self.plot_network(self.similarity_matrix, self.molecules, self.smiles_list)
    
    def clear(self):
        self.axes.clear()
        self.axes.set_facecolor(BLACK)
        self.molecules = None
        self.similarity_matrix = None
        self.smiles_list = None
        self.node_positions = {}
        self.edge_list = []
        self.edge_lines = []
        self.draw()

class TonimotoCalculator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle('Tanimoto Similarity Calculator')
        self.setGeometry(100, 100, 1600, 1000)
        
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: {BLACK};
                color: {YELLOW};
            }}
            QPushButton {{
                background-color: {YELLOW};
                color: {BLACK};
                border: none;
                padding: 8px 15px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 11pt;
            }}
            QPushButton:hover {{
                background-color: {DARK_YELLOW};
            }}
            QTextEdit {{
                background-color: {LIGHT_BLACK};
                color: {WHITE};
                border: 2px solid {YELLOW};
                border-radius: 5px;
                padding: 5px;
                font-size: 11pt;
            }}
            QLabel {{
                color: {YELLOW};
                font-size: 11pt;
            }}
            QSlider::groove:horizontal {{
                border: 2px solid {YELLOW};
                height: 10px;
                background: {LIGHT_BLACK};
                margin: 2px 0;
                border-radius: 5px;
            }}
            QSlider::handle:horizontal {{
                background: {YELLOW};
                border: 2px solid {DARK_YELLOW};
                width: 20px;
                margin: -2px 0;
                border-radius: 5px;
            }}
            QGroupBox {{
                background-color: {LIGHT_BLACK};
                color: {YELLOW};
                border: 2px solid {YELLOW};
                border-radius: 5px;
                margin-top: 15px;
                padding: 10px;
            }}
        """)
        
        # Create a scroll area for the main content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background-color: {BLACK};
            }}
            QScrollBar:vertical {{
                border: none;
                background: {LIGHT_BLACK};
                width: 12px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background: {YELLOW};
                min-height: 20px;
                border-radius: 6px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                border: none;
                background: none;
            }}
            QScrollBar:horizontal {{
                border: none;
                background: {LIGHT_BLACK};
                height: 12px;
                margin: 0px;
            }}
            QScrollBar::handle:horizontal {{
                background: {YELLOW};
                min-width: 20px;
                border-radius: 6px;
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                border: none;
                background: none;
            }}
        """)
        
        central_widget = QWidget()
        scroll_area.setWidget(central_widget)
        self.setCentralWidget(scroll_area)
        
        # Main vertical layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Top input section
        top_section = QWidget()
        top_layout = QHBoxLayout(top_section)
        top_layout.setSpacing(20)
        
        # Input group
        input_group = QGroupBox("Input Data")
        input_layout = QVBoxLayout()
        
        smiles_label = QLabel("SMILE Sequences:")
        self.smiles_input = QTextEdit()
        self.smiles_input.setPlaceholderText("Enter each SMILE sequence on a new line.\nExample:\nCCO\nCCN\nCCC")
        self.smiles_input.setMinimumHeight(100)
        
        input_layout.addWidget(smiles_label)
        input_layout.addWidget(self.smiles_input)
        input_group.setLayout(input_layout)
        
        # Control group
        control_group = QGroupBox("Controls")
        control_layout = QVBoxLayout()
        
        button_layout = QHBoxLayout()
        self.calculate_button = QPushButton("Calculate Similarity")
        self.clear_button = QPushButton("Clear")
        self.calculate_button.clicked.connect(self.calculate_tonimoto)
        self.clear_button.clicked.connect(self.clear_data)
        
        button_layout.addWidget(self.calculate_button)
        button_layout.addWidget(self.clear_button)
        
        threshold_layout = QHBoxLayout()
        threshold_label = QLabel("Similarity Threshold:")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(70)
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        self.threshold_value = QLabel("0.70")
        
        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_value)
        
        control_layout.addLayout(button_layout)
        control_layout.addLayout(threshold_layout)
        control_group.setLayout(control_layout)
        
        top_layout.addWidget(input_group, stretch=2)
        top_layout.addWidget(control_group, stretch=1)
        
        # Bottom section with network graph and molecule preview
        bottom_section = QWidget()
        bottom_layout = QHBoxLayout(bottom_section)
        bottom_layout.setSpacing(20)
        
        # Network graph
        graph_group = QGroupBox("Network Graph")
        graph_layout = QVBoxLayout()
        graph_layout.setContentsMargins(10, 10, 10, 10)
        
        self.network_canvas = SimpleNetworkCanvas()
        graph_layout.addWidget(self.network_canvas)
        
        graph_group.setLayout(graph_layout)
        
        # Molecule preview
        self.preview_widget = MoleculePreviewWidget()
        self.network_canvas.preview_widget = self.preview_widget
        
        # Add widgets to bottom layout
        bottom_layout.addWidget(graph_group, stretch=3)
        bottom_layout.addWidget(self.preview_widget, stretch=1)
        
        # Add sections to main layout
        main_layout.addWidget(top_section)
        main_layout.addWidget(bottom_section, stretch=1)
    
    def calculate_tonimoto(self):
        smiles_text = self.smiles_input.toPlainText()
        if not smiles_text:
            QMessageBox.warning(self, "Error", "Please enter SMILE sequences.")
            return
        
        smiles_list = [s.strip() for s in smiles_text.split('\n') if s.strip()]
        
        if len(smiles_list) < 2:
            QMessageBox.warning(self, "Error", "At least two SMILE sequences are required.")
            return
        
        valid_mols = []
        valid_smiles = []
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_mols.append(mol)
                valid_smiles.append(smiles)
        
        if len(valid_mols) < 2:
            QMessageBox.warning(self, "Error", "At least two valid SMILE sequences are required.")
            return
        
        n = len(valid_mols)
        similarity_matrix = np.zeros((n, n))
        
        fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024) for mol in valid_mols]
        
        for i in range(n):
            for j in range(i, n):
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
        
        self.network_canvas.plot_network(similarity_matrix, valid_mols, valid_smiles)
    
    def update_threshold(self, value):
        threshold = value / 100.0
        self.threshold_value.setText(f"{threshold:.2f}")
        self.network_canvas.update_threshold(threshold)
    
    def clear_data(self):
        self.smiles_input.clear()
        self.network_canvas.clear()
        self.preview_widget.update_molecules(None, None)
        self.threshold_slider.setValue(70)
        self.threshold_value.setText("0.70")

def main():
    app = QApplication(sys.argv)
    window = TonimotoCalculator()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 