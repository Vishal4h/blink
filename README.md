# 👁️ Blink Project

> A real-time eye blink detection and monitoring application using **MediaPipe Face Mesh**, **OpenCV**, and **Eye Aspect Ratio (EAR)** analysis.

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Mesh-orange?style=for-the-badge)

---

## ✨ Features

- 📷 Real-time webcam access
- 👤 Face landmark detection
- 👁️ Eye landmark extraction
- 📐 Eye Aspect Ratio (EAR) calculation
- 👀 Individual blink detection
- 📊 Blink rate calculation
- ⚙️ Adaptive calibration
- 🚨 Low blink-rate warning
- 🔊 Sound alert
- 🖥️ Transparent HUD
- ⚡ Real-time monitoring

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Webcam access and image processing |
| MediaPipe Face Mesh | Facial landmark detection |
| NumPy | Numerical calculations |
| EAR | Eye closure and blink detection |

---

## 🔄 How It Works

1. The application accesses the computer's webcam.
2. **MediaPipe Face Mesh** detects facial landmarks.
3. Eye landmarks are extracted from the detected face.
4. **Eye Aspect Ratio (EAR)** is calculated to identify eye closure.
5. `BlinkDetector` identifies individual blinks.
6. The blink rate is calculated in **blinks per minute**.
7. Adaptive calibration adjusts the blink detection threshold.
8. If the blink rate drops below the configured threshold, a warning and sound alert are triggered.
9. A transparent **HUD (Heads-Up Display)** displays real-time monitoring information.
10. The system continuously monitors the user's blink activity in real time.

---

## 🧮 Eye Aspect Ratio

The Eye Aspect Ratio is used to determine whether the eye is open or closed.

```text
Eye Landmarks
      ↓
Vertical Eye Distance
      ↓
Horizontal Eye Distance
      ↓
Calculate EAR
      ↓
Compare EAR with Threshold
      ↓
Open / Closed Eye
      ↓
Blink Detection
```

---

## 👁️ Blink Detection

```text
Webcam
   ↓
MediaPipe Face Mesh
   ↓
Eye Landmarks
   ↓
EAR Calculation
   ↓
BlinkDetector
   ↓
Blink Detected
```

---

## 📊 Blink Rate

The detected blinks are converted into a blink rate measured in:

```text
Blinks Per Minute (BPM)
```

The blink rate is continuously updated while the webcam is active.

---

## ⚙️ Adaptive Calibration

Adaptive calibration adjusts the blink detection threshold based on the user's eye characteristics and current conditions.

This helps improve detection reliability and reduce false blink detections.

---

## 🚨 Alert System

```text
Blink Rate
     ↓
Compare With Threshold
     ↓
┌─────────────────────┐
│ Blink Rate Normal   │
│ Continue Monitoring │
└─────────────────────┘

          OR

┌─────────────────────┐
│ Blink Rate Too Low  │
└──────────┬──────────┘
           ↓
    Warning Triggered
           ↓
      Sound Alert
           ↓
      HUD Warning
```

---

## 🖥️ Transparent HUD

The application displays a transparent **Heads-Up Display (HUD)** over the webcam feed.

Example:

```text
┌─────────────────────────────┐
│       BLINK MONITOR         │
│                             │
│  Blink Rate: 18 BPM         │
│  Status: Monitoring         │
│  Calibration: Complete      │
│                             │
└─────────────────────────────┘
```

The HUD can display:

- Current blink rate
- Detection status
- Calibration status
- Warning status
- Eye detection information

---

## 📁 Project Structure

```text
Blink-Project/
│
├── main.py
├── blink_detector.py
├── requirements.txt
├── README.md
│
├── assets/
│   └── sounds/
│
└── screenshots/
```

---

## 📦 Installation

### Clone the Repository

```bash
git clone https://github.com/imvishal004/imvishal004.git
cd Blink-Project
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
python main.py
```

Make sure your webcam is connected and accessible.

---

## 📋 Requirements

- Python 3.x
- Working webcam
- OpenCV
- MediaPipe
- NumPy
- Required Python dependencies

---

## 🎯 Use Cases

- 👁️ Blink monitoring
- 🧠 Human-computer interaction
- 📚 Computer vision learning
- 🧪 Eye-tracking experiments
- 🚗 Driver alertness research
- 🖥️ Real-time webcam applications

---

## 🚀 Future Improvements

- [ ] Advanced fatigue detection
- [ ] Yawning detection
- [ ] Head pose estimation
- [ ] Drowsiness score
- [ ] Blink-rate graphs
- [ ] Session history
- [ ] User-specific calibration profiles
- [ ] Improved low-light detection
- [ ] Customizable HUD
- [ ] Export monitoring statistics

---

## ⚠️ Limitations

Detection accuracy may be affected by:

- Poor lighting
- Low webcam quality
- Face partially outside the frame
- Glasses or reflective lenses
- Extreme head angles
- Multiple faces
- Temporary landmark detection failures

---

## 🔐 Privacy

The application processes webcam frames locally for real-time blink detection.

The webcam feed is not intended to be uploaded to a remote server.

---

## 👨‍💻 Developer

**Vishal**

Computer Science & Engineering — Artificial Intelligence & Machine Learning

---

## 📜 License

This project is licensed under the **MIT License**.

---

## ⭐ Support

If you find this project useful:

⭐ Star the repository  
🍴 Fork the repository  
🐛 Report issues  
💡 Suggest improvements

---

## 🙌 Acknowledgements

This project uses:

- [MediaPipe](https://github.com/google-ai-edge/mediapipe)
- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)

---

## 📬 Contact

For suggestions, issues, or contributions, open an issue or submit a pull request.

---

**Built with ❤️ using Python and Computer Vision.**
