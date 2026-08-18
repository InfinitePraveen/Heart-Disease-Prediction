# CardioSense — Heart Disease Risk Console

A Flask web app that serves the ensemble model trained in [`../ml-project`](../ml-project),
styled as a futuristic patient-monitor console: a live sweeping ECG trace, a sectioned
"patient chart" intake form, and a reactive risk readout — built as an interview-ready
demo of taking a notebook model to a deployed, interactive product.

![status](https://img.shields.io/badge/status-demo-3DFFB0)

## Design

The interface is built around a real bedside cardiac monitor: a continuously sweeping
ECG trace as the hero element, an intake form organized into "channels" the way a monitor
groups vitals, and a live readout panel whose waveform speed, jitter, and color react to
the model's predicted risk band (calm green trace for low risk, fast irregular red trace
for high risk). Typeface: IBM Plex Sans for headings/body, IBM Plex Mono for every numeric
readout — the same pairing instrument panels use to keep data legible at a glance.

## How it works

1. `templates/index.html` renders the intake form directly from `model/model_metadata.json`
   — add/remove a clinical field there and the form updates automatically.
2. On submit, `static/js/script.js` POSTs the 12 raw clinical inputs to `/api/predict`.
3. `app.py` recreates the exact feature engineering from the training notebook
   (heart-rate reserve, age×cholesterol interaction, hypertensive-range flag), runs the
   saved voting ensemble, and returns a probability, risk band, and the top contributing
   factors for that specific patient.
4. The readout panel animates the risk number, swaps the ECG trace's rhythm/color to match
   the risk band, and lists the contributing factors as chips.

## Run it

```bash
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:5000**. Use "Load sample chart" for a one-click demo.

> ⚠️ Educational/portfolio demo only, trained on 303 records from the UCI Cleveland
> dataset. Not a medical device and not intended for real diagnosis.

## Author

**Praveen**
GitHub: [github.com/InfinitePraveen](https://github.com/InfinitePraveen)
LinkedIn: [linkedin.com/in/infinitepraveen](https://www.linkedin.com/in/infinitepraveen/)
