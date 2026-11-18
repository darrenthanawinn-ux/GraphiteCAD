# GraphiteCAD — Acoustic Pen-to-Vector Engine

> **What if you could turn any ordinary $0.50 wooden pencil and a sheet of paper into a wireless digital stylus using nothing more than your laptop's built-in microphone and real-time acoustic physics?**

**GraphiteCAD** is a browser-based Digital Signal Processing (DSP) and vector kinematics engine built for **IdeaLab Hacks**. With zero cameras, zero Bluetooth hardware, and zero external sensors, it captures the micro-acoustic friction of graphite scratching against paper and translates those sound waves into live 2D CAD vector drawings at 60fps.

---

## ✨ Key Features

- **Real-Time Acoustic Vector Drawing:** Detects `PEN DOWN` vs. `PEN UP` states purely from friction energy, mapping stroke pressure to dynamic line weight (`1.5px`–`6px`) and acoustic pitch to stroke velocity and curvature.
- **Tap-to-Reposition Impulse Detection:** Tapping your pencil tip on the desk generates a sharp transient spike (`>2.2x` rolling energy average) that lifts the virtual pen and jumps the cursor to a new anchor point.
- **3-Second Room Noise Calibration:** Automatically samples ambient room acoustics and sets a dynamic signal threshold above your environment's noise floor.
- **Live FFT Frequency Oscilloscope:** Visualizes raw frequency bins from `0 Hz` to `16 kHz` in real time, highlighting the active `3 kHz – 12 kHz` graphite friction band.
- **Built-In Acoustic Simulator Mode:** Synthesizes procedural pencil-scratch and tap telemetry so anyone can inspect and test the DSP pipeline even in a noisy environment or without a pencil handy.
- **One-Click SVG Export:** Exports generated vector strokes as a clean, scalable `.svg` file ready for CAD or vector design software.

---

## 🔬 How the Acoustic DSP Works

1. **Raw Microphone Capture:** Disables browser voice-chat processing (`echoCancellation: false`, `noiseSuppression: false`, `autoGainControl: false`) so high-frequency graphite friction isn't filtered out as background noise.
2. **High-Pass Filtering:** Routes raw audio through a Web Audio `BiquadFilterNode` (`highpass` at `2500 Hz`) to strip low-frequency room hum, fans, and human speech.
3. **Spectral Feature Extraction (60fps):** Uses an `AnalyserNode` (`fftSize = 2048`) to compute:
   - **Friction Energy (RMS):** Measures acoustic intensity inside the `3 kHz – 12 kHz` graphite cleavage band.
   - **Spectral Centroid ($f_c$):** Calculates the frequency-weighted center of mass to steer vector curvature and velocity:
     $$f_c = \frac{\sum_{k} f_k \cdot A_k}{\sum_{k} A_k}$$
   - **Transient Spike Detection:** Identifies sharp mechanical impacts (pencil taps) to trigger pen-up breaks.

---

### Prerequisites
- Node.js (v18+ recommended)
- A modern browser (Chrome, Edge, or Firefox) with microphone access
