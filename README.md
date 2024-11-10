Voice and Face Detection System
This project is a real-time detection system that tracks voice activity and facial features, built to assist in scenarios such as online proctoring, security monitoring, and user verification. By detecting multiple voices and monitoring face and eye movements, it helps identify unusual activities or unauthorized presence, ensuring integrity in monitored environments.

Features
Dual Voice Detection: Uses WebRTC's Voice Activity Detection (webrtcvad) to identify multiple voices, helping to detect unauthorized background sounds.
Face and Eye Tracking: Leverages OpenCV’s Haar Cascade classifiers to detect faces and eyes, tracking their positions in real-time.
Real-Time Alerts: Provides live feedback on detected activities, such as multiple voices or unusual eye movements, indicating potential cheating or unauthorized presence.
User Interface: Simple GUI built with tkinter for easy interaction and visualization of video feed and detection alerts.
Technology Stack
Audio Processing: pyaudio, webrtcvad
Computer Vision: opencv-python
GUI Framework: tkinter
Image Processing: Pillow for rendering frames in the GUI
Installation
Requirements
Python 3.7+
pyaudio
webrtcvad
opencv-python
numpy
Pillow
tkinter (included with Python on most systems)
Additional Requirements for Windows
Some dependencies may require setup tools:

cmake: Needed to build certain packages on Windows.
dlib (optional): Installable via pip for extended face feature tracking, though not mandatory for basic functionality.
Installation Steps
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/voice_face_detection.git
cd voice_face_detection
Install the dependencies:

bash
Copy code
pip install -r requirements.txt
If you encounter issues on Windows, make sure cmake is installed and add it to your PATH.

Usage
Run the detection system:

bash
Copy code
python voice_face_detection.py
Start Detection: Click "Start Detection" to begin monitoring voice and face activity.

Stop Detection: Click "Stop Detection" to halt the detection process.

Project Structure
voice_face_detection.py: Main script containing audio and video detection functions.
README.md: Project documentation.
requirements.txt: List of dependencies for quick installation.
Example Use Cases
Remote Proctoring: Detect multiple voices or suspicious eye movements during online exams.
Security Monitoring: Track unauthorized entry in restricted zones.
User Verification: Validate user presence and behavior in secured access areas.
License
This project is open-source under the MIT License. Feel free to use, modify, and distribute the code with proper attribution.

Contributing
Contributions are welcome! Please open an issue or submit a pull request to improve this project.

Contact
For questions or feedback, contact your-email@example.com.

