/* CardioSense — console interactions
   - Two independently animated ECG sweep traces (hero background + live readout)
   - Wires the vitals form to POST /api/predict and renders the response
*/

(() => {
  "use strict";

  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---------------------------------------------------------------------
  // ECG waveform generator — sum-of-Gaussians approximation of a PQRST beat
  // ---------------------------------------------------------------------
  function ecgValue(phase) {
    const g = (x, mu, sigma, h) => h * Math.exp(-((x - mu) ** 2) / (2 * sigma * sigma));
    let v = 0;
    v += g(phase, 0.12, 0.020, 0.12);   // P wave
    v -= g(phase, 0.26, 0.008, 0.14);   // Q dip
    v += g(phase, 0.29, 0.009, 1.00);   // R spike
    v -= g(phase, 0.32, 0.010, 0.28);   // S dip
    v += g(phase, 0.50, 0.050, 0.28);   // T wave
    return v;
  }

  class ECGTrace {
    constructor(canvas, opts = {}) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      this.bpm = opts.bpm ?? 70;
      this.amplitude = opts.amplitude ?? 0.7;
      this.color = opts.color ?? "#3DFFB0";
      this.jitter = opts.jitter ?? 0; // 0..1, adds beat-to-beat irregularity
      this.lineWidth = opts.lineWidth ?? 2;
      this.speed = opts.speed ?? 55; // px/sec sweep speed
      this.t = 0;
      this.buffer = [];
      this._resize();
      window.addEventListener("resize", () => this._resize());
      this._last = performance.now();
      this._raf = null;
    }

    _resize() {
      const rect = this.canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      this.width = Math.max(1, Math.floor(rect.width));
      this.height = Math.max(1, Math.floor(rect.height));
      this.canvas.width = this.width * dpr;
      this.canvas.height = this.height * dpr;
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      if (this.buffer.length === 0) {
        this.buffer = new Array(this.width).fill(0);
      }
    }

    setParams(opts = {}) {
      Object.assign(this, opts);
    }

    start() {
      if (this._raf) return;
      this._last = performance.now();
      const loop = (now) => {
        const dt = Math.min((now - this._last) / 1000, 0.05);
        this._last = now;
        this._step(dt);
        this._draw();
        this._raf = requestAnimationFrame(loop);
      };
      this._raf = requestAnimationFrame(loop);
    }

    stop() {
      if (this._raf) cancelAnimationFrame(this._raf);
      this._raf = null;
    }

    _step(dt) {
      const period = 60 / this.bpm;
      const jitterAmt = this.jitter * (Math.sin(this.t * 3.1) * 0.15 + (Math.random() - 0.5) * 0.1);
      const effPeriod = Math.max(0.25, period * (1 + jitterAmt));
      this.t += dt;
      const phase = (this.t % effPeriod) / effPeriod;
      const raw = ecgValue(phase) * this.amplitude;
      const noise = (Math.random() - 0.5) * 0.015;
      const px = Math.round(this.speed * dt);
      for (let i = 0; i < Math.max(1, px); i++) {
        this.buffer.push(raw + noise);
      }
      while (this.buffer.length > this.width) this.buffer.shift();
    }

    _draw() {
      const ctx = this.ctx;
      ctx.clearRect(0, 0, this.width, this.height);
      const midY = this.height / 2;
      const scaleY = this.height * 0.42;

      ctx.beginPath();
      ctx.lineWidth = this.lineWidth;
      ctx.strokeStyle = this.color;
      ctx.shadowColor = this.color;
      ctx.shadowBlur = 6;
      ctx.lineJoin = "round";

      const len = this.buffer.length;
      for (let i = 0; i < len; i++) {
        const x = this.width - (len - i);
        const y = midY - this.buffer[i] * scaleY;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // sweep head glow
      const headX = this.width - 1;
      const headY = midY - (this.buffer[len - 1] || 0) * scaleY;
      ctx.beginPath();
      ctx.fillStyle = this.color;
      ctx.arc(headX, headY, 2.6, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // ---------------------------------------------------------------------
  // Boot traces
  // ---------------------------------------------------------------------
  const heroCanvas = document.getElementById("hero-ecg");
  const heroTrace = new ECGTrace(heroCanvas, {
    bpm: 68, amplitude: 0.55, color: "#3DFFB0", lineWidth: 1.6, speed: 46, jitter: 0.02,
  });

  const readoutCanvas = document.getElementById("readout-ecg");
  const readoutTrace = new ECGTrace(readoutCanvas, {
    bpm: 62, amplitude: 0.35, color: "#4CC9F0", lineWidth: 2, speed: 60, jitter: 0.01,
  });

  if (!prefersReducedMotion) {
    heroTrace.start();
    readoutTrace.start();
  } else {
    heroTrace._draw();
    readoutTrace._draw();
  }

  // ---------------------------------------------------------------------
  // Monitor clock
  // ---------------------------------------------------------------------
  const clockEl = document.getElementById("monitor-clock");
  function tickClock() {
    const now = new Date();
    clockEl.textContent = now.toLocaleTimeString("en-GB", { hour12: false });
  }
  tickClock();
  setInterval(tickClock, 1000);

  // ---------------------------------------------------------------------
  // Form submission → /api/predict
  // ---------------------------------------------------------------------
  const form = document.getElementById("vitals-form");
  const runBtn = document.getElementById("run-btn");
  const statusDot = document.getElementById("status-dot");
  const statusText = document.getElementById("status-text");

  const riskValueEl = document.getElementById("risk-value");
  const riskTagEl = document.getElementById("risk-tag");
  const bpmValueEl = document.getElementById("bpm-value");
  const verdictValueEl = document.getElementById("verdict-value");
  const factorsListEl = document.getElementById("factors-list");

  const FACTOR_LABELS = {
    age: "Age", sex: "Sex", cp: "Chest Pain Type", trestbps: "Resting BP",
    chol: "Cholesterol", restecg: "Resting ECG", thalach: "Max Heart Rate",
    exang: "Exercise Angina", oldpeak: "ST Depression", slope: "ST Slope",
    ca: "Vessels Colored", thal: "Thalassemia", hr_reserve: "Heart-Rate Reserve",
    age_chol_interaction: "Age × Cholesterol", high_risk_bp: "Hypertensive Range",
  };

  function collectPayload() {
    const data = {};
    form.querySelectorAll("[data-field]").forEach((el) => {
      data[el.name] = Number(el.value);
    });
    return data;
  }

  function applyRiskVisuals(tone) {
    const map = {
      low: { color: "#3DFFB0", bpm: 66, amplitude: 0.4, jitter: 0.01 },
      moderate: { color: "#FFB238", bpm: 92, amplitude: 0.55, jitter: 0.12 },
      high: { color: "#FF4D5E", bpm: 118, amplitude: 0.75, jitter: 0.28 },
    };
    const p = map[tone] || map.low;
    readoutTrace.setParams(p);
    statusDot.classList.toggle("alert", tone === "high");
    statusText.textContent = tone === "high" ? "ELEVATED RISK DETECTED" : "MONITORING ACTIVE";
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    runBtn.disabled = true;
    runBtn.textContent = "Analyzing…";
    statusText.textContent = "RUNNING DIAGNOSTIC…";

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(collectPayload()),
      });
      if (!res.ok) throw new Error((await res.json()).error || "Prediction failed");
      const result = await res.json();

      // Animate the risk number counting up
      const target = result.risk_percent;
      let current = 0;
      const startTime = performance.now();
      const duration = 700;
      function animateCount(now) {
        const p = Math.min(1, (now - startTime) / duration);
        current = target * p;
        riskValueEl.textContent = current.toFixed(1);
        if (p < 1) requestAnimationFrame(animateCount);
        else riskValueEl.textContent = target.toFixed(1);
      }
      requestAnimationFrame(animateCount);

      riskTagEl.textContent = result.risk_label;
      riskTagEl.className = "risk-tag " + result.risk_tone;
      riskValueEl.style.color =
        result.risk_tone === "high" ? "#FF4D5E" : result.risk_tone === "moderate" ? "#FFB238" : "#3DFFB0";

      const thalachInput = form.querySelector('[name="thalach"]');
      bpmValueEl.innerHTML = `${thalachInput.value} <small>bpm</small>`;

      verdictValueEl.textContent = result.prediction === 1 ? "DISEASE LIKELY" : "NO DISEASE";
      verdictValueEl.style.color = result.prediction === 1 ? "#FF4D5E" : "#3DFFB0";

      factorsListEl.innerHTML = "";
      result.top_factors.forEach((f) => {
        const chip = document.createElement("span");
        chip.className = "factor-chip";
        chip.textContent = FACTOR_LABELS[f] || f;
        factorsListEl.appendChild(chip);
      });

      applyRiskVisuals(result.risk_tone);
    } catch (err) {
      statusText.textContent = "ERROR — SEE CONSOLE";
      console.error(err);
      alert("Something went wrong running the diagnostic: " + err.message);
    } finally {
      runBtn.disabled = false;
      runBtn.innerHTML = '<span class="run-btn-icon">⌁</span> Run Diagnostic';
    }
  });

  // ---------------------------------------------------------------------
  // Sample chart loader — quick demo path
  // ---------------------------------------------------------------------
  const sampleBtn = document.getElementById("sample-btn");
  const SAMPLE = {
    age: 52, sex: 1, cp: 1, trestbps: 128, chol: 205, restecg: 1,
    thalach: 184, exang: 0, oldpeak: 0.0, slope: 2, ca: 0, thal: 2,
  };
  sampleBtn.addEventListener("click", () => {
    Object.entries(SAMPLE).forEach(([name, value]) => {
      const el = form.querySelector(`[name="${name}"]`);
      if (el) el.value = value;
    });
  });
})();
