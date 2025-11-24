# Raspberry ENV-ML — TinyML inference on Raspberry Pi

**Monsoon Semester Project II — Hardware Architecture for Deep Learning (EC6252E)**

> A compact TinyML project demonstrating audio-event detection (fire alarm detector) running on a Raspberry Pi using a TensorFlow Lite model.

---

## Project overview

This repository contains a small TinyML inference pipeline built for a Raspberry Pi class project. It includes a trained `model.tflite`, example audio samples, a training notebook (`Tiny_ML_Train.ipynb`), a simple inference script, and a `requirements.txt` to reproduce the runtime environment.

**Key goals**

* Demonstrate audio-based event detection with a TFLite model on resource-constrained hardware.
* Provide a minimal training notebook showing how the model was produced (or how to reproduce / fine-tune it).
* Provide scripts and sample data to test inference on a Raspberry Pi.

---

## Repository structure

```
├── LICENSE
├── README.md                # (this file)
├── Tiny_ML_Train.ipynb      # Notebook: data prep + model training / conversion to TFLite
├── model.tflite             # Trained TensorFlow Lite model for audio detection
├── requirements.txt         # Python dependencies for running inference / dev
├── new_file.py              # Example inference / utility script
├── fire-alarm-414915.wav    # Sample audio (positive class)
├── output_fire_alarm.wav    # Processed / output audio
├── output_bg.wav            # Background / negative sample
```

> Note: File names listed are based on the repository contents. If you rename files locally, update the commands below accordingly.

---

## Quick start — Run inference on Raspberry Pi (recommended)

1. **Prepare the Pi**

   * Raspbian / Raspberry Pi OS (64-bit recommended) up-to-date.
   * Python 3.9+ (or the version that matches `requirements.txt`).

2. **Clone the repo**

```bash
git clone https://github.com/nav-jk/raspberry-env-ml-nav.git
cd raspberry-env-ml-nav
```

3. **Create a virtual environment and install dependencies**

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Run the example inference script**

```bash
python new_file.py --input fire-alarm-414915.wav --model model.tflite
```

> If `new_file.py` expects a different CLI, open the file and adapt the command. The script name is intentionally generic — replace with your preferred inference script.

---

## About the model

* The repository contains a TensorFlow Lite model (`model.tflite`) intended for audio classification (fire-alarm detection). The model is small and suitable for edge inference on a Raspberry Pi or microcontroller with TFLite runtime.
* If you want to inspect or test the model on desktop first, install `tflite-runtime` or full `tensorflow` and run a small script that loads the model and prints output tensors.

Minimal tester snippet:

```python
import numpy as np
import tflite_runtime.interpreter as tflite

interpreter = tflite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Build a dummy input with the expected shape
input_shape = input_details[0]['shape']
dummy = np.zeros(input_shape, dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], dummy)
interpreter.invoke()
print(interpreter.get_tensor(output_details[0]['index']))
```

---

## Training / Notebook

Open `Tiny_ML_Train.ipynb` to see the training pipeline used to produce the model. The notebook should contain data-preprocessing steps (feature extraction — e.g., spectrogram or MFCC), model definition, training loop, and conversion to TFLite. If you plan to re-train or fine-tune:

* Verify dataset locations and update paths inside the notebook.
* Confirm sample rate and input preprocessing (e.g., number of mel bins / frame length) — the TFLite model expects the same preprocessing used during training.

---

## Tips for improving or extending this repo

* **Document `new_file.py`**: Add a `--help` or top-of-file docstring that describes CLI arguments and expected input shapes. Example: `--input`, `--model`, `--sample-rate`, `--frame-length`.
* **Add unit tests**: small tests to check model loading and that the expected input shape matches the preprocessing pipeline.
* **Make an inference wrapper**: create a `run_inference.py` that:

  * loads the TFLite model,
  * performs audio preprocessing (resample, normalize, convert to spectrogram/MFCC),
  * runs inference and returns class probabilities and timestamps (if needed).
* **Benchmarking script**: measure latency and RAM usage on a Raspberry Pi (use `time` or `perf` and `psutil`).
* **Add a sample `systemd` service or simple daemon** to run real-time audio capture (e.g., via `arecord`/`sounddevice`) and call the inference wrapper.

---

## Dependencies

See `requirements.txt` for the exact packages used. Typical packages for this workflow include:

* `numpy`
* `scipy`
* `tflite-runtime` or `tensorflow` (for desktop testing)
* `soundfile` or `pydub` or `librosa` for audio I/O and preprocessing

If `requirements.txt` uses `tensorflow` and you only need inference on the Pi, prefer `tflite-runtime` to save space.

---

## License

This project is distributed under the **MIT License**. See the `LICENSE` file for details.

---
