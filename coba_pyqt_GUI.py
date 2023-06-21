import sys
import serial
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QLineEdit, QPushButton
from coba_pyqt_main import main

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Processing Program")
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout()

        # Create labels and text inputs for each variable
        self.device_label = QLabel("Device:")
        self.device_input = QLineEdit()

        self.userid_label = QLabel("UserID:")
        self.userid_input = QLineEdit()

        self.duration_label = QLabel("Recording Duration:")
        self.duration_input = QLineEdit()

        self.process_button = QPushButton("Process")
        self.process_button.clicked.connect(self.process_data)

        layout.addWidget(self.device_label)
        layout.addWidget(self.device_input)
        layout.addWidget(self.userid_label)
        layout.addWidget(self.userid_input)
        layout.addWidget(self.duration_label)
        layout.addWidget(self.duration_input)
        layout.addWidget(self.process_button)

        self.central_widget.setLayout(layout)

    def process_data(self):
        device = self.device_input.text()
        userid = int(self.userid_input.text())
        recording_duration = int(self.duration_input.text())

        serial_port = serial.Serial(device, baudrate=9600, timeout=1)
        main(userid, serial_port, recording_duration)

def run_application():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    run_application()