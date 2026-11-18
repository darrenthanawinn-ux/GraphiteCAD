import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Activity,
  ChevronDown,
  ChevronUp,
  Download,
  FlaskConical,
  Mic,
  MicOff,
  PenLine,
  Target,
  Trash2,
} from 'lucide-react';

/* ════════════════════════════ CONSTANTS ════════════════════════════ */
const CYAN = '#00F0FF';
const AMBER = '#FFB020';
const MM = 3.7795; // px per mm (96 dpi)
const FFT = 2048;
const BINS = FFT / 2;
const DB_FLOOR = -100; // friction-band RMS dB mapped to 0
const DB_SPAN = 70; // …and DB_FLOOR + DB_SPAN mapped to 1
const SPEC_MAX = 16000;
const BAND_LO = 3000;
const BAND_HI = 12000;
const CAL_SECONDS = 3;

const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const rnd = (a: number, b: number) => a + Math.random() * (b - a);
const wrapPi = (a: number) => {
  while (a > Math.PI) a -= Math.PI * 2;
  while (a < -Math.PI) a += Math.PI * 2;
  return a;
};

/* ════════════════════════════ TYPES ════════════════════════════ */
type Pt = { x: number; y: number; w: number };
type Stroke = Pt[];

interface SimState {
  phase: 'gap' | 'stroke';
  until: number;
  level: number;
  scratchAt: number;
  tapUntil: number;
  cBase: number;
}

interface Engine {
  ctx: AudioContext | null;
  analyser: AnalyserNode | null;
  stream: MediaStream | null;
  freqDb: Float32Array;
  timeBuf: Float32Array;
  strokes: Stroke[];
  cur: Stroke | null;
  x: number;
  y: number;
  theta: number;
  v: number;
  vel: number;
  init: boolean;
  penDown: boolean;
  belowSince: number;
  ignoreUntil: number;
  lastTap: number;
  eS: number;
  cS: number;
  cMean: number;
  prevC: number;
  ampHist: number[];
  prevAmp: number;
  baseline: number;
  ambientAmp: number;
  calib: { start: number; en: number[]; am: number[] } | null;
  taps: number;
  sim: SimState;
  ripple: { x: number; y: number; t: number } | null;
  w: number;
  h: number;
  grid: HTMLCanvasElement | null;
  layer: HTMLCanvasElement | null;
  lc: CanvasRenderingContext2D | null;
  lastTs: number;
  lastUi: number;
}

interface Tele {
  energy: number;
  db: number;
  centroid: number;
  velocity: number;
  taps: number;
  penDown: boolean;
  x: number;
  y: number;
  thr: number;
  live: boolean;
  calib: number | null;
  paths: number;
}

const makeEngine = (): Engine => ({
  ctx: null,
  analyser: null,
  stream: null,
  freqDb: new Float32Array(BINS).fill(-140),
  timeBuf: new Float32Array(FFT),
  strokes: [],
  cur: null,
  x: 0,
  y: 0,
  theta: Math.random() * Math.PI * 2,
  v: 0,
  vel: 0,
  init: false,
  penDown: false,
  belowSince: 0,
  ignoreUntil: 0,
  lastTap: -1,
  eS: 0,
  cS: 7000,
  cMean: 7000,
  prevC: 7000,
  ampHist: [],
  prevAmp: 0,
  baseline: 0.08,
  ambientAmp: 0.002,
  calib: null,
  taps: 0,
  sim: { phase: 'gap', until: 0.4, level: 0.7, scratchAt: 0, tapUntil: 0, cBase: 6500 },
  ripple: null,
  w: 0,
  h: 0,
  grid: null,
  layer: null,
  lc: null,
  lastTs: 0,
  lastUi: 0,
});

/* ════════════════════════════ DSP HELPERS ════════════════════════════ */

/** Friction-band (3–12 kHz) RMS energy and power-weighted spectral centroid. */
function analyze(db: Float32Array, sr: number) {
  const binHz = sr / FFT;
  const lo = Math.ceil(BAND_LO / binHz);
  const hi = Math.min(BINS - 1, Math.floor(BAND_HI / binHz));
  let sq = 0;
  let num = 0;
  let den = 0;
  let n = 0;
  for (let k = lo; k <= hi; k++) {
    const m = Math.pow(10, Math.max(db[k], -140) / 20);
    const pw = m * m;
    sq += pw;
    num += k * binHz * pw;
    den += pw;
    n++;
  }
  const rms = Math.sqrt(sq / Math.max(1, n));
  const d = 20 * Math.log10(rms + 1e-12);
  return { energy: clamp((d - DB_FLOOR) / DB_SPAN, 0, 1), db: d, centroid: den > 0 ? num / den : 0 };
}

/** Procedural pencil-scratch + tap telemetry. Fills e.freqDb and returns time-domain RMS amplitude. */
function simStep(e: Engine, t: number): number {
  const s = e.sim;
  if (t >= s.until) {
    if (s.phase === 'gap') {
      s.phase = 'stroke';
      s.until = t + rnd(0.9, 2.4);
      s.level = rnd(0.55, 0.95);
      const tap = Math.random() < 0.7;
      s.tapUntil = tap ? t + 0.045 : 0;
      s.scratchAt = t + (tap ? 0.2 : 0.02);
      s.cBase = rnd(4800, 8600);
    } else {
      s.phase = 'gap';
      s.until = t + rnd(0.35, 0.9);
      s.tapUntil = 0;
    }
  }
  let env = 0;
  if (s.phase === 'stroke' && t >= s.scratchAt) {
    const a = clamp((t - s.scratchAt) / 0.12, 0, 1);
    const r = clamp((s.until - t) / 0.15, 0, 1);
    env = s.level * Math.min(a, r) * clamp(0.85 + 0.15 * Math.sin(t * 41) * Math.sin(t * 13.7) + rnd(-0.06, 0.06), 0, 1.2);
  }
  const tap = t < s.tapUntil ? 1 : 0;
  const c = clamp(s.cBase + 2400 * Math.sin(t * 1.7) + 900 * Math.sin(t * 4.1 + 1) + rnd(-300, 300), 3300, 11500);
  const binHz = 48000 / FFT;
  for (let k = 0; k < BINS; k++) {
    const f = k * binHz;
    let m = 1.6e-5 * (1 + Math.random() * 0.7) * (f < 2500 ? 0.08 : 1);
    if (env > 0) {
      const z = (f - c) / 2400;
      m += env * 0.004 * Math.exp(-0.5 * z * z) * rnd(0.7, 1.25);
    }
    if (tap) m += 0.008 * (f < 2500 ? 0.05 : 1) * rnd(0.5, 1);
    e.freqDb[k] = 20 * Math.log10(m + 1e-9);
  }
  return 0.0012 * rnd(1, 1.5) + env * 0.012 + tap * 0.08;
}

function endStroke(e: Engine) {
  if (e.cur && e.cur.length < 2) {
    const i = e.strokes.indexOf(e.cur);
    if (i >= 0) e.strokes.splice(i, 1);
  }
  e.cur = null;
}

/* ════════════════════════════ RENDER HELPERS ════════════════════════════ */

function fit(cv: HTMLCanvasElement) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = cv.clientWidth;
  const h = cv.clientHeight;
  const pw = Math.round(w * dpr);
  const ph = Math.round(h * dpr);
  let ch = false;
  if (cv.width !== pw || cv.height !== ph) {
    cv.width = pw;
    cv.height = ph;
    ch = true;
  }
  return { w, h, dpr, ch };
}

function drawGrid(g: CanvasRenderingContext2D, w: number, h: number) {
  g.fillStyle = '#0A0C10';
  g.fillRect(0, 0, w, h);
  const vg = g.createRadialGradient(w / 2, h / 2, Math.min(w, h) * 0.2, w / 2, h / 2, Math.max(w, h) * 0.75);
  vg.addColorStop(0, 'rgba(20,34,52,0.35)');
  vg.addColorStop(1, 'rgba(0,0,0,0.35)');
  g.fillStyle = vg;
  g.fillRect(0, 0, w, h);
  const minor = 5 * MM;
  const major = 25 * MM;
  g.lineWidth = 1;
  g.strokeStyle = 'rgba(70,110,150,0.11)';
  g.beginPath();
  for (let x = 0; x <= w; x += minor) {
    g.moveTo(Math.round(x) + 0.5, 0);
    g.lineTo(Math.round(x) + 0.5, h);
  }
  for (let y = 0; y <= h; y += minor) {
    g.moveTo(0, Math.round(y) + 0.5);
    g.lineTo(w, Math.round(y) + 0.5);
  }
  g.stroke();
  g.strokeStyle = 'rgba(80,150,200,0.24)';
  g.beginPath();
  for (let x = 0; x <= w; x += major) {
    g.moveTo(Math.round(x) + 0.5, 0);
    g.lineTo(Math.round(x) + 0.5, h);
  }
  for (let y = 0; y <= h; y += major) {
    g.moveTo(0, Math.round(y) + 0.5);
    g.lineTo(w, Math.round(y) + 0.5);
  }
  g.stroke();
  g.fillStyle = 'rgba(120,150,180,0.55)';
  g.font = '9px ui-monospace, Menlo, Consolas, monospace';
  g.textBaseline = 'top';
  for (let i = 1; i * major < w; i++) g.fillText(String(i * 25), i * major + 3, 3);
  for (let i = 1; i * major < h; i++) g.fillText(String(i * 25), 3, i * major + 3);
  g.fillText('mm', 3, 3);
}

function drawSeg(lc: CanvasRenderingContext2D, a: Pt, b: Pt) {
  lc.lineCap = 'round';
  lc.lineJoin = 'round';
  lc.beginPath();
  lc.moveTo(a.x, a.y);
  lc.lineTo(b.x, b.y);
  lc.strokeStyle = 'rgba(0,240,255,0.14)';
  lc.lineWidth = b.w * 3.4;
  lc.stroke();
  lc.strokeStyle = 'rgba(0,240,255,0.32)';
  lc.lineWidth = b.w * 1.9;
  lc.stroke();
  lc.strokeStyle = CYAN;
  lc.lineWidth = b.w;
  lc.stroke();
}

function redrawLayer(e: Engine) {
  if (!e.lc || !e.layer) return;
  e.lc.clearRect(0, 0, e.w, e.h);
  for (const s of e.strokes) for (let i = 1; i < s.length; i++) drawSeg(e.lc, s[i - 1], s[i]);
}

function drawSpectrum(
  c: CanvasRenderingContext2D,
  w: number,
  h: number,
  db: Float32Array,
  sr: number,
  thr: number,
  centroid: number,
  live: boolean
) {
  const padL = 30;
  const padB = 16;
  const padT = 6;
  const pw = w - padL - 4;
  const ph = h - padB - padT;
  const yOf = (d: number) => padT + ph - clamp((d + 110) / 90, 0, 1) * ph;
  const xOf = (f: number) => padL + (f / SPEC_MAX) * pw;
  c.fillStyle = '#070A0E';
  c.fillRect(0, 0, w, h);

  // friction zone
  const zx = xOf(BAND_LO);
  const zw = xOf(BAND_HI) - zx;
  const zg = c.createLinearGradient(0, padT, 0, padT + ph);
  zg.addColorStop(0, 'rgba(255,176,32,0.16)');
  zg.addColorStop(1, 'rgba(255,176,32,0.03)');
  c.fillStyle = zg;
  c.fillRect(zx, padT, zw, ph);
  c.strokeStyle = 'rgba(255,176,32,0.45)';
  c.setLineDash([3, 3]);
  c.lineWidth = 1;
  c.beginPath();
  c.moveTo(zx + 0.5, padT);
  c.lineTo(zx + 0.5, padT + ph);
  c.moveTo(zx + zw + 0.5, padT);
  c.lineTo(zx + zw + 0.5, padT + ph);
  c.stroke();
  c.setLineDash([]);

  // grid + labels
  c.font = '9px ui-monospace, Menlo, Consolas, monospace';
  c.textBaseline = 'middle';
  c.textAlign = 'right';
  for (const d of [-30, -50, -70, -90]) {
    const y = Math.round(yOf(d)) + 0.5;
    c.strokeStyle = 'rgba(80,110,140,0.18)';
    c.beginPath();
    c.moveTo(padL, y);
    c.lineTo(w - 4, y);
    c.stroke();
    c.fillStyle = 'rgba(120,140,165,0.7)';
    c.fillText(String(d), padL - 4, y);
  }
  c.textAlign = 'center';
  c.textBaseline = 'alphabetic';
  for (const f of [0, 4000, 8000, 12000, 16000]) {
    c.fillStyle = 'rgba(120,140,165,0.7)';
    c.fillText(f === 0 ? '0' : f / 1000 + 'k', clamp(xOf(f), 12, w - 10), h - 4);
  }
  c.fillStyle = 'rgba(255,176,32,0.85)';
  c.textAlign = 'left';
  c.fillText('GRAPHITE FRICTION ZONE 3–12 kHz', zx + 5, padT + 10);

  // spectrum
  const binHz = sr / FFT;
  const kMax = Math.min(BINS - 1, Math.floor(SPEC_MAX / binHz));
  c.beginPath();
  for (let k = 0; k <= kMax; k++) {
    const x = padL + ((k * binHz) / SPEC_MAX) * pw;
    const y = yOf(db[k]);
    if (k === 0) c.moveTo(x, y);
    else c.lineTo(x, y);
  }
  c.lineJoin = 'round';
  c.strokeStyle = CYAN;
  c.lineWidth = 1.2;
  c.stroke();
  c.lineTo(padL + pw, padT + ph);
  c.lineTo(padL, padT + ph);
  c.closePath();
  const fg = c.createLinearGradient(0, padT, 0, padT + ph);
  fg.addColorStop(0, 'rgba(0,240,255,0.32)');
  fg.addColorStop(1, 'rgba(0,240,255,0.02)');
  c.fillStyle = fg;
  c.fill();

  // threshold
  const ty = yOf(DB_FLOOR + thr * DB_SPAN);
  c.strokeStyle = '#FF4D6D';
  c.setLineDash([6, 4]);
  c.lineWidth = 1;
  c.beginPath();
  c.moveTo(padL, ty);
  c.lineTo(w - 4, ty);
  c.stroke();
  c.setLineDash([]);
  c.fillStyle = '#FF4D6D';
  c.textAlign = 'right';
  c.fillText('GATE ' + (DB_FLOOR + thr * DB_SPAN).toFixed(0) + ' dB', w - 6, ty - 5);

  // centroid
  if (live && centroid > 0) {
    const cx = xOf(centroid);
    c.strokeStyle = 'rgba(230,240,255,0.7)';
    c.setLineDash([2, 3]);
    c.beginPath();
    c.moveTo(cx, padT);
    c.lineTo(cx, padT + ph);
    c.stroke();
    c.setLineDash([]);
  }
  c.strokeStyle = 'rgba(80,110,140,0.5)';
  c.strokeRect(padL + 0.5, padT + 0.5, pw, ph);
}

function buildSvg(strokes: Stroke[], w: number, h: number) {
  const f = (n: number) => n.toFixed(2);
  const paths = strokes
    .filter((s) => s.length > 1)
    .map((s, idx) => {
      let d = `M${f(s[0].x)} ${f(s[0].y)}`;
      for (let i = 1; i < s.length - 1; i++) {
        const mx = (s[i].x + s[i + 1].x) / 2;
        const my = (s[i].y + s[i + 1].y) / 2;
        d += ` Q${f(s[i].x)} ${f(s[i].y)} ${f(mx)} ${f(my)}`;
      }
      const last = s[s.length - 1];
      d += ` L${f(last.x)} ${f(last.y)}`;
      const avg = s.reduce((a, p) => a + p.w, 0) / s.length;
      return `    <path id="stroke-${idx + 1}" d="${d}" stroke-width="${f(avg)}"/>`;
    })
    .join('\n');
  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">
  <title>GraphiteCAD acoustic vector export</title>
  <defs>
    <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="2.2" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <rect width="100%" height="100%" fill="#0A0C10"/>
  <g fill="none" stroke="${CYAN}" stroke-linecap="round" stroke-linejoin="round" filter="url(#glow)">
${paths}
  </g>
</svg>
`;
}

/* ════════════════════════════ UI PIECES ════════════════════════════ */

function Slider(props: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  fmt: (v: number) => string;
  onChange: (v: number) => void;
}) {
  return (
    <label className="gc-slider">
      <span className="gc-slider-top">
        <span>{props.label}</span>
        <b>{props.fmt(props.value)}</b>
      </span>
      <input
        type="range"
        min={props.min}
        max={props.max}
        step={props.step ?? 1}
        value={props.value}
        onChange={(ev) => props.onChange(Number(ev.target.value))}
      />
    </label>
  );
}

function Card(props: { label: string; value: string; sub: string; pct: number; tone?: string }) {
  return (
    <div className="gc-card">
      <div className="gc-card-label">{props.label}</div>
      <div className="gc-card-value" style={props.tone ? { color: props.tone } : undefined}>
        {props.value}
      </div>
      <div className="gc-card-sub">{props.sub}</div>
      <div className="gc-bar">
        <i style={{ width: `${clamp(props.pct, 0, 1) * 100}%`, background: props.tone || CYAN }} />
      </div>
    </div>
  );
}

/* ════════════════════════════ APP ════════════════════════════ */

const INITIAL_TELE: Tele = {
  energy: 0,
  db: -140,
  centroid: 0,
  velocity: 0,
  taps: 0,
  penDown: false,
  x: 0,
  y: 0,
  thr: 0.26,
  live: false,
  calib: null,
  paths: 0,
};

export default function App() {
  const E = useRef<Engine>(null as unknown as Engine);
  if (!E.current) E.current = makeEngine();
  const mainRef = useRef<HTMLCanvasElement>(null);
  const specRef = useRef<HTMLCanvasElement>(null);

  const [micOn, setMicOn] = useState(false);
  const [micErr, setMicErr] = useState<string | null>(null);
  const [sim, setSim] = useState(false);
  const [calibrated, setCalibrated] = useState(false);
  const [msg, setMsg] = useState('Start the microphone or enable Simulator Mode to begin.');
  const [tele, setTele] = useState<Tele>(INITIAL_TELE);
  const [gate, setGate] = useState(35);
  const [curv, setCurv] = useState(100);
  const [smooth, setSmooth] = useState(50);
  const [notesOpen, setNotesOpen] = useState(false);

  const P = useRef({ sim, gate: gate / 100, curv: curv / 100, smooth: smooth / 100 });
  P.current = { sim, gate: gate / 100, curv: curv / 100, smooth: smooth / 100 };

  /* ───────── microphone ───────── */
  const stopMic = useCallback(() => {
    const e = E.current;
    e.stream?.getTracks().forEach((tr) => tr.stop());
    if (e.ctx && e.ctx.state !== 'closed') e.ctx.close().catch(() => undefined);
    e.stream = null;
    e.ctx = null;
    e.analyser = null;
    setMicOn(false);
  }, []);

  const startMic = useCallback(async () => {
    const e = E.current;
    setMicErr(null);
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('getUserMedia unavailable (use https or localhost)');
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      });
      const AC: typeof AudioContext = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const ctx = new AC();
      await ctx.resume();
      const src = ctx.createMediaStreamSource(stream);
      const hp = ctx.createBiquadFilter();
      hp.type = 'highpass';
      hp.frequency.value = 2500;
      hp.Q.value = 0.707;
      const an = ctx.createAnalyser();
      an.fftSize = FFT;
      an.smoothingTimeConstant = 0.35;
      const mute = ctx.createGain();
      mute.gain.value = 0; // keeps the graph pulled without echoing the mic
      src.connect(hp);
      hp.connect(an);
      an.connect(mute);
      mute.connect(ctx.destination);
      e.stream = stream;
      e.ctx = ctx;
      e.analyser = an;
      setMicOn(true);
      setCalibrated(false);
      setMsg('Microphone live. Press Calibrate (3s) with the room quiet, then draw with a pencil near the mic.');
    } catch (err) {
      const m = err instanceof Error ? err.message : String(err);
      setMicErr(m);
      setMsg('Microphone error: ' + m + '. Simulator Mode works without a mic.');
    }
  }, []);

  /* ───────── actions ───────── */
  const calibrate = () => {
    const e = E.current;
    if (!micOn && !sim) {
      setMsg('Calibration needs a live source: start the microphone or enable Simulator Mode.');
      return;
    }
    e.calib = { start: -1, en: [], am: [] };
    setMsg('Calibrating: stay quiet for 3 seconds…');
  };

  const clearCanvas = () => {
    const e = E.current;
    e.strokes = [];
    e.cur = null;
    if (e.penDown) e.penDown = false;
    redrawLayer(e);
    setMsg('Canvas cleared.');
  };

  const exportSVG = () => {
    const e = E.current;
    const n = e.strokes.filter((s) => s.length > 1).length;
    if (!n) {
      setMsg('Nothing to export yet: draw some vector paths first.');
      return;
    }
    const svg = buildSvg(e.strokes, Math.round(e.w), Math.round(e.h));
    const blob = new Blob([svg], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `graphitecad-${new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)}.svg`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1500);
    setMsg(`Exported ${n} vector path${n === 1 ? '' : 's'} to SVG.`);
  };

  /* ───────── 60 fps DSP + render loop ───────── */
  useEffect(() => {
    let raf = 0;
    const tick = (ts: number) => {
      raf = requestAnimationFrame(tick);
      const e = E.current;
      const p = P.current;
      const mc = mainRef.current;
      const sc = specRef.current;
      if (!mc || !sc) return;
      const t = ts / 1000;
      const dt = clamp(e.lastTs ? t - e.lastTs : 0.016, 0.001, 0.05);
      e.lastTs = t;

      // canvas sizing / offscreen layers
      const m = fit(mc);
      if (m.w < 2 || m.h < 2) return;
      if (m.ch || !e.grid || !e.layer) {
        e.w = m.w;
        e.h = m.h;
        const g = document.createElement('canvas');
        g.width = mc.width;
        g.height = mc.height;
        const gc = g.getContext('2d') as CanvasRenderingContext2D;
        gc.setTransform(m.dpr, 0, 0, m.dpr, 0, 0);
        drawGrid(gc, m.w, m.h);
        e.grid = g;
        const l = document.createElement('canvas');
        l.width = mc.width;
        l.height = mc.height;
        const lc = l.getContext('2d') as CanvasRenderingContext2D;
        lc.setTransform(m.dpr, 0, 0, m.dpr, 0, 0);
        e.layer = l;
        e.lc = lc;
        redrawLayer(e);
        if (!e.init) {
          e.x = m.w / 2;
          e.y = m.h / 2;
          e.init = true;
        } else {
          e.x = clamp(e.x, 20, m.w - 20);
          e.y = clamp(e.y, 20, m.h - 20);
        }
      }

      // ── acquire spectrum + amplitude
      let live = false;
      let amp = 0;
      let sr = 48000;
      if (p.sim) {
        amp = simStep(e, t);
        live = true;
      } else if (e.analyser && e.ctx) {
        e.analyser.getFloatFrequencyData(e.freqDb);
        e.analyser.getFloatTimeDomainData(e.timeBuf);
        let s = 0;
        for (let i = 0; i < e.timeBuf.length; i++) s += e.timeBuf[i] * e.timeBuf[i];
        amp = Math.sqrt(s / e.timeBuf.length);
        sr = e.ctx.sampleRate;
        live = true;
      } else {
        e.freqDb.fill(-140);
      }
      const met = live ? analyze(e.freqDb, sr) : { energy: 0, db: -140, centroid: 0 };

      // smoothing
      const a = clamp(1 - p.smooth, 0.04, 1);
      e.eS = lerp(e.eS, met.energy, a);
      if (live && met.centroid > 0) e.cS = lerp(e.cS, met.centroid, a);

      // ── calibration (3 s ambient noise sampling)
      if (e.calib && !live) e.calib = null;
      let calibP: number | null = null;
      if (e.calib) {
        const cb = e.calib;
        if (cb.start < 0) cb.start = t;
        const el = t - cb.start;
        cb.en.push(met.energy);
        cb.am.push(amp);
        if (el >= CAL_SECONDS) {
          const mean = (arr: number[]) => arr.reduce((x, y) => x + y, 0) / Math.max(1, arr.length);
          const sd = (arr: number[], mu: number) => Math.sqrt(mean(arr.map((x) => (x - mu) * (x - mu))));
          const me = mean(cb.en);
          const ma = mean(cb.am);
          e.baseline = clamp(me + 2 * sd(cb.en, me) + 0.02, 0.03, 0.85);
          e.ambientAmp = clamp(ma + 2 * sd(cb.am, ma), 0.0005, 0.05);
          e.calib = null;
          e.ignoreUntil = t + 0.3;
          setCalibrated(true);
          setMsg(`Calibrated: room floor ${(me * 100).toFixed(0)}% energy, gate base ${(e.baseline * 100).toFixed(0)}%.`);
        } else {
          calibP = el / CAL_SECONDS;
        }
      }

      const thr = clamp(e.baseline + 0.04 + p.gate * 0.35, 0.02, 0.97);

      // ── tap transient + kinematic pen engine
      if (live && !e.calib) {
        const hist = e.ampHist;
        const avg = hist.length ? hist.reduce((x, y) => x + y, 0) / hist.length : amp;
        const tapMin = Math.max(0.006, e.ambientAmp * 4);
        if (amp > 2.2 * avg && amp > tapMin && amp > e.prevAmp * 1.8 && t - e.lastTap > 0.2) {
          e.lastTap = t;
          e.taps++;
          endStroke(e);
          e.penDown = false;
          e.belowSince = 0;
          e.ignoreUntil = t + 0.18;
          const ang = rnd(0, Math.PI * 2);
          const dist = rnd(50, 140);
          e.x = clamp(e.x + Math.cos(ang) * dist, 30, e.w - 30);
          e.y = clamp(e.y + Math.sin(ang) * dist, 30, e.h - 30);
          e.theta = rnd(0, Math.PI * 2);
          e.ripple = { x: e.x, y: e.y, t };
        }
        hist.push(amp);
        if (hist.length > 30) hist.shift();
        e.prevAmp = amp;

        const active = e.eS > thr;
        if (!e.penDown) {
          if (active && t > e.ignoreUntil) {
            e.penDown = true;
            e.belowSince = 0;
            e.cMean = e.cS;
            e.prevC = e.cS;
            e.cur = [{ x: e.x, y: e.y, w: 1.5 }];
            e.strokes.push(e.cur);
          }
        } else if (e.eS < thr * 0.85) {
          if (!e.belowSince) e.belowSince = t;
          else if (t - e.belowSince > 0.08) {
            endStroke(e);
            e.penDown = false;
            e.belowSince = 0;
          }
        } else {
          e.belowSince = 0;
        }

        if (e.penDown && e.cur) {
          const eN = clamp((e.eS - thr) / (1 - thr), 0, 1);
          const vNorm = clamp((e.cS - 3000) / 9000, 0, 1);
          const vT = (60 + 520 * Math.pow(vNorm, 1.15)) * (0.65 + 0.35 * eN);
          e.v = lerp(e.v, vT, 0.15);
          const dC = e.cS - e.prevC;
          e.prevC = e.cS;
          e.cMean = lerp(e.cMean, e.cS, 0.02);
          const dTheta =
            (clamp(dC / 50, -2, 2) * 2.0 + clamp((e.cS - e.cMean) / 2000, -1.5, 1.5) * 2.5) * p.curv;
          e.theta += dTheta * dt;
          const mg = 60;
          if (e.x < mg || e.x > e.w - mg || e.y < mg || e.y > e.h - mg) {
            const des = Math.atan2(e.h / 2 - e.y, e.w / 2 - e.x);
            e.theta += clamp(wrapPi(des - e.theta), -1, 1) * 4 * dt;
          }
          e.x = clamp(e.x + Math.cos(e.theta) * e.v * dt, 4, e.w - 4);
          e.y = clamp(e.y + Math.sin(e.theta) * e.v * dt, 4, e.h - 4);
          const last = e.cur[e.cur.length - 1];
          if (Math.hypot(e.x - last.x, e.y - last.y) >= 1.2) {
            const pt: Pt = { x: e.x, y: e.y, w: 1.5 + 4.5 * eN };
            e.cur.push(pt);
            if (e.lc) drawSeg(e.lc, last, pt);
          }
        }
      } else if (e.penDown) {
        endStroke(e);
        e.penDown = false;
      }
      e.vel = lerp(e.vel, e.penDown ? e.v / MM : 0, 0.2);

      // ── render: main canvas
      const c = mc.getContext('2d') as CanvasRenderingContext2D;
      c.setTransform(m.dpr, 0, 0, m.dpr, 0, 0);
      c.clearRect(0, 0, m.w, m.h);
      if (e.grid) c.drawImage(e.grid, 0, 0, m.w, m.h);
      if (e.layer) c.drawImage(e.layer, 0, 0, m.w, m.h);

      if (e.ripple) {
        const age = t - e.ripple.t;
        if (age < 0.6) {
          c.strokeStyle = `rgba(255,176,32,${(1 - age / 0.6) * 0.9})`;
          c.lineWidth = 1.5;
          c.beginPath();
          c.arc(e.ripple.x, e.ripple.y, 8 + age * 90, 0, Math.PI * 2);
          c.stroke();
        } else e.ripple = null;
      }

      const col = e.penDown ? CYAN : '#8CA0B8';
      c.save();
      c.strokeStyle = col;
      c.globalAlpha = e.penDown ? 0.3 : 0.18;
      c.lineWidth = 1;
      c.setLineDash([4, 6]);
      c.beginPath();
      c.moveTo(e.x + 0.5, 0);
      c.lineTo(e.x + 0.5, m.h);
      c.moveTo(0, e.y + 0.5);
      c.lineTo(m.w, e.y + 0.5);
      c.stroke();
      c.restore();
      c.strokeStyle = col;
      c.lineWidth = 1.4;
      c.beginPath();
      c.arc(e.x, e.y, 10, 0, Math.PI * 2);
      c.moveTo(e.x - 16, e.y);
      c.lineTo(e.x - 5, e.y);
      c.moveTo(e.x + 5, e.y);
      c.lineTo(e.x + 16, e.y);
      c.moveTo(e.x, e.y - 16);
      c.lineTo(e.x, e.y - 5);
      c.moveTo(e.x, e.y + 5);
      c.lineTo(e.x, e.y + 16);
      c.stroke();
      if (e.penDown) {
        c.fillStyle = CYAN;
        c.beginPath();
        c.arc(e.x, e.y, 2.5, 0, Math.PI * 2);
        c.fill();
      }
      c.font = '10px ui-monospace, Menlo, Consolas, monospace';
      c.textBaseline = 'top';
      c.fillStyle = col;
      const lx = e.x + 16 + 120 > m.w ? e.x - 130 : e.x + 16;
      c.fillText(`X ${(e.x / MM).toFixed(1)}  Y ${(e.y / MM).toFixed(1)} mm`, lx, e.y + 14);
      if (!e.strokes.length && !live) {
        c.fillStyle = 'rgba(140,160,185,0.6)';
        c.font = '13px ui-monospace, Menlo, Consolas, monospace';
        c.textAlign = 'center';
        c.fillText('Start Mic or enable Simulator Mode', m.w / 2, m.h / 2 + 40);
        c.textAlign = 'left';
      }

      // ── render: spectrum
      const s = fit(sc);
      if (s.w > 2) {
        const sx = sc.getContext('2d') as CanvasRenderingContext2D;
        sx.setTransform(s.dpr, 0, 0, s.dpr, 0, 0);
        drawSpectrum(sx, s.w, s.h, e.freqDb, sr, thr, live ? e.cS : 0, live);
      }

      // ── throttled React telemetry (~15 Hz)
      if (ts - e.lastUi > 66) {
        e.lastUi = ts;
        setTele({
          energy: e.eS,
          db: met.db,
          centroid: live ? e.cS : 0,
          velocity: e.vel,
          taps: e.taps,
          penDown: e.penDown,
          x: e.x / MM,
          y: e.y / MM,
          thr,
          live,
          calib: calibP,
          paths: e.strokes.length,
        });
      }
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf);
      stopMic();
    };
  }, [stopMic]);

  /* ───────── status pill ───────── */
  let pillText = 'OFFLINE';
  let pillCls = 'off';
  if (micErr && !sim) {
    pillText = 'MIC ERROR';
    pillCls = 'err';
  }
  if (sim) {
    pillText = 'SIMULATOR LIVE';
    pillCls = 'live';
  } else if (micOn) {
    pillText = calibrated ? 'MIC LIVE · CALIBRATED' : 'MIC LIVE · UNCALIBRATED';
    pillCls = calibrated ? 'live' : 'warn';
  }
  if (tele.calib !== null) {
    pillText = `CALIBRATING ${(CAL_SECONDS * (1 - tele.calib)).toFixed(1)}s`;
    pillCls = 'cal';
  }

  return (
    <div className="gc-root">
      <style>{CSS}</style>

      <header className="gc-head">
        <div className="gc-logo">
          <PenLine size={18} color={CYAN} />
          <span className="gc-logo-main">GraphiteCAD</span>
          <span className="gc-logo-sub">// ACOUSTIC DSP</span>
        </div>
        <div className={`gc-pill ${pillCls}`}>
          <i />
          {pillText}
        </div>
        <div className="gc-actions">
          <button className={`gc-btn ${micOn ? 'on' : ''}`} onClick={() => (micOn ? stopMic() : startMic())}>
            {micOn ? <MicOff size={14} /> : <Mic size={14} />}
            {micOn ? 'Stop Mic' : 'Start Mic'}
          </button>
          <button className="gc-btn" onClick={calibrate} disabled={tele.calib !== null}>
            <Target size={14} />
            Calibrate (3s)
          </button>
          <button
            className={`gc-btn gc-switch ${sim ? 'on' : ''}`}
            role="switch"
            aria-checked={sim}
            onClick={() => {
              setSim((v) => !v);
              setMsg(!sim ? 'Simulator Mode: procedural pencil-scratch and tap telemetry.' : 'Simulator Mode off.');
            }}
          >
            <Activity size={14} />
            Simulator Mode
            <span className="gc-knob">
              <i />
            </span>
          </button>
          <button className="gc-btn" onClick={clearCanvas}>
            <Trash2 size={14} />
            Clear Canvas
          </button>
          <button className="gc-btn gc-primary" onClick={exportSVG}>
            <Download size={14} />
            Export SVG
          </button>
        </div>
      </header>

      <main className="gc-main">
        <section className="gc-stage">
          <canvas ref={mainRef} />
          <div className={`gc-hud gc-pen ${tele.penDown ? 'down' : 'up'}`}>
            <span className="dot" />
            {tele.penDown ? 'PEN DOWN' : 'PEN UP'}
          </div>
          <div className="gc-hud gc-xy">
            X {tele.x.toFixed(1)} · Y {tele.y.toFixed(1)} mm · {tele.paths} path{tele.paths === 1 ? '' : 's'}
          </div>
          <div className="gc-hud gc-msg">{msg}</div>
        </section>

        <aside className="gc-side">
          <div className="gc-panel">
            <div className="gc-panel-title">FFT Frequency Spectrum · 0–16 kHz</div>
            <canvas ref={specRef} className="gc-spec" />
          </div>

          <div className="gc-grid2">
            <Card
              label="Friction Energy"
              value={`${(tele.energy * 100).toFixed(0)}%`}
              sub={`${tele.db > -139 ? tele.db.toFixed(1) : '−∞'} dB · gate ${(tele.thr * 100).toFixed(0)}%`}
              pct={tele.energy}
              tone={tele.energy > tele.thr ? CYAN : undefined}
            />
            <Card
              label="Spectral Centroid"
              value={`${Math.round(tele.centroid)} Hz`}
              sub={tele.centroid > 0 ? (tele.centroid > 7500 ? 'fast stroke' : 'slow shading') : 'no signal'}
              pct={(tele.centroid - 3000) / 9000}
            />
            <Card
              label="Stroke Velocity"
              value={`${tele.velocity.toFixed(1)} mm/s`}
              sub={tele.penDown ? 'pen down' : 'pen lifted'}
              pct={tele.velocity / 155}
              tone={AMBER}
            />
            <Card label="Tap Count" value={String(tele.taps)} sub="transients > 2.2× avg" pct={(tele.taps % 10) / 10} tone="#FF4D6D" />
          </div>

          <div className="gc-panel gc-controls">
            <div className="gc-panel-title">DSP Controls</div>
            <Slider label="Noise Gate" value={gate} min={0} max={100} fmt={(v) => `${v}%`} onChange={setGate} />
            <Slider label="Curvature Gain" value={curv} min={0} max={300} fmt={(v) => `${(v / 100).toFixed(2)}×`} onChange={setCurv} />
            <Slider label="Smoothing" value={smooth} min={0} max={95} fmt={(v) => `${v}%`} onChange={setSmooth} />
          </div>

          <div className="gc-panel">
            <button className="gc-collapse" onClick={() => setNotesOpen((o) => !o)} aria-expanded={notesOpen}>
              <FlaskConical size={14} />
              Experiment &amp; Physics Notes
              {notesOpen ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            </button>
            {notesOpen && (
              <div className="gc-notes">
                <h4>Why graphite sings at 3–12 kHz</h4>
                <p>
                  Pencil “lead” is a composite of graphite and clay. Graphite is built from stacked graphene sheets bonded weakly
                  to each other (van der Waals), so the crystallites cleave easily along their basal planes. Dragging the tip over
                  paper shears those crystallites off and fractures grains against the cellulose fibres of the paper tooth.
                </p>
                <p>
                  Each cleavage or stick-slip release is a micro-impulse that radiates a short broadband acoustic emission. Thousands
                  per second sum into a hiss whose energy concentrates in the 3–12 kHz band, well above speech and room hum, which
                  is why the pipeline high-passes at 2.5 kHz.
                </p>
                <h4>How the signal maps to geometry</h4>
                <p>
                  <b>Friction energy</b> tracks pressure and contact: it gates the pen and sets stroke width (1.5–6 px).{' '}
                  <b>Spectral centroid</b> rises with drag speed, so it sets velocity, and its changes steer the heading to curve the
                  path. A <b>tap transient</b> (&gt; 2.2× the rolling average) lifts the pen and relocates the cursor.
                </p>
                <h4>Getting a clean capture</h4>
                <p>
                  Place the mic 5–10 cm from the paper, work in a quiet room and run Calibrate (3s) first. Echo cancellation, noise
                  suppression and auto-gain are disabled so the friction band is not filtered away. The mapping is a heuristic
                  instrument, not calibrated metrology: expect to tune Noise Gate and Smoothing for your mic and paper.
                </p>
              </div>
            )}
          </div>
        </aside>
      </main>
    </div>
  );
}

/* ════════════════════════════ STYLES ════════════════════════════ */
const CSS = `
*{box-sizing:border-box}
html,body,#root{height:100%;margin:0;background:#0A0C10}
.gc-root{--bg:#0A0C10;--panel:#0E1218;--line:#1B2431;--tx:#C9D5E3;--dim:#64748B;--cy:#00F0FF;--am:#FFB020;--rd:#FF4D6D;--gn:#39FFA0;
  height:100%;display:flex;flex-direction:column;background:var(--bg);color:var(--tx);
  font-family:'JetBrains Mono','SF Mono',ui-monospace,Menlo,Consolas,monospace;font-size:12px;overflow:hidden}
.gc-head{display:flex;align-items:center;gap:14px;padding:9px 14px;border-bottom:1px solid var(--line);background:linear-gradient(#0E131B,#0A0C10);flex-wrap:wrap}
.gc-logo{display:flex;align-items:center;gap:8px;white-space:nowrap}
.gc-logo-main{font-weight:700;font-size:15px;letter-spacing:.02em;color:#E8F4FA}
.gc-logo-sub{color:var(--cy);font-size:12px;letter-spacing:.08em}
.gc-pill{display:flex;align-items:center;gap:7px;padding:4px 10px;border:1px solid var(--line);border-radius:999px;font-size:11px;letter-spacing:.04em;color:var(--dim);background:#0B0F15;white-space:nowrap}
.gc-pill i{width:7px;height:7px;border-radius:50%;background:var(--dim)}
.gc-pill.live{color:var(--gn);border-color:rgba(57,255,160,.35)}.gc-pill.live i{background:var(--gn);box-shadow:0 0 8px var(--gn)}
.gc-pill.warn{color:var(--am);border-color:rgba(255,176,32,.4)}.gc-pill.warn i{background:var(--am);box-shadow:0 0 8px var(--am)}
.gc-pill.cal{color:var(--cy);border-color:rgba(0,240,255,.4)}.gc-pill.cal i{background:var(--cy);animation:gcpulse .7s infinite alternate}
.gc-pill.err{color:var(--rd);border-color:rgba(255,77,109,.4)}.gc-pill.err i{background:var(--rd)}
@keyframes gcpulse{from{opacity:.3}to{opacity:1}}
.gc-actions{margin-left:auto;display:flex;gap:8px;flex-wrap:wrap}
.gc-btn{display:inline-flex;align-items:center;gap:7px;height:30px;padding:0 11px;background:#101620;color:var(--tx);border:1px solid #243041;border-radius:5px;font:inherit;font-size:11.5px;cursor:pointer;transition:border-color .15s,background .15s,color .15s}
.gc-btn:hover:not(:disabled){border-color:var(--cy);color:#fff;background:#121B27}
.gc-btn:disabled{opacity:.45;cursor:default}
.gc-btn:focus-visible,.gc-collapse:focus-visible,.gc-slider input:focus-visible{outline:2px solid var(--cy);outline-offset:2px}
.gc-btn.on{border-color:rgba(0,240,255,.6);color:var(--cy);background:rgba(0,240,255,.07)}
.gc-primary{border-color:rgba(0,240,255,.5);color:#04141A;background:var(--cy);font-weight:700}
.gc-primary:hover:not(:disabled){background:#5CF6FF;color:#04141A;border-color:#5CF6FF}
.gc-knob{width:26px;height:14px;border-radius:999px;background:#1A2330;border:1px solid #2A3647;position:relative;display:inline-block}
.gc-knob i{position:absolute;top:1px;left:1px;width:10px;height:10px;border-radius:50%;background:var(--dim);transition:transform .18s,background .18s}
.gc-switch.on .gc-knob{background:rgba(0,240,255,.18);border-color:rgba(0,240,255,.6)}
.gc-switch.on .gc-knob i{transform:translateX(12px);background:var(--cy)}
.gc-main{flex:1;min-height:0;display:grid;grid-template-columns:70% 30%}
.gc-stage{position:relative;min-width:0;min-height:0;border-right:1px solid var(--line);background:#0A0C10}
.gc-stage canvas{position:absolute;inset:0;width:100%;height:100%;display:block}
.gc-hud{position:absolute;pointer-events:none;font-size:11px;letter-spacing:.04em}
.gc-pen{top:12px;left:14px;display:flex;align-items:center;gap:8px;padding:5px 11px;border-radius:4px;border:1px solid var(--line);background:rgba(10,12,16,.82);font-weight:700}
.gc-pen .dot{width:8px;height:8px;border-radius:50%}
.gc-pen.down{color:var(--cy);border-color:rgba(0,240,255,.5)}.gc-pen.down .dot{background:var(--cy);box-shadow:0 0 10px var(--cy)}
.gc-pen.up{color:#8CA0B8}.gc-pen.up .dot{background:#5C6B80}
.gc-xy{top:14px;right:14px;color:#8CA0B8;background:rgba(10,12,16,.75);padding:4px 8px;border-radius:4px}
.gc-msg{left:14px;bottom:12px;right:14px;color:#7F90A6;background:rgba(10,12,16,.72);padding:5px 9px;border-radius:4px;border-left:2px solid var(--cy)}
.gc-side{min-width:0;overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:12px;background:#0B0E13}
.gc-panel{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:10px}
.gc-panel-title{font-size:10.5px;letter-spacing:.06em;color:var(--dim);margin-bottom:8px}
.gc-spec{width:100%;height:200px;display:block;border-radius:3px}
.gc-grid2{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.gc-card{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:9px 10px;min-width:0}
.gc-card-label{font-size:10px;color:var(--dim);letter-spacing:.04em}
.gc-card-value{font-size:19px;font-weight:700;color:#E6F1F8;margin:4px 0 2px;white-space:nowrap}
.gc-card-sub{font-size:10px;color:#6F8097;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.gc-bar{height:3px;background:#161E2A;border-radius:2px;margin-top:7px;overflow:hidden}
.gc-bar i{display:block;height:100%;border-radius:2px;transition:width .08s linear}
.gc-controls{display:flex;flex-direction:column;gap:12px}
.gc-controls .gc-panel-title{margin-bottom:0}
.gc-slider{display:flex;flex-direction:column;gap:6px;font-size:11px}
.gc-slider-top{display:flex;justify-content:space-between;color:#9FB0C4}
.gc-slider-top b{color:var(--cy);font-weight:600}
.gc-slider input{-webkit-appearance:none;appearance:none;width:100%;height:4px;background:#1B2533;border-radius:2px;outline:none}
.gc-slider input::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;background:var(--cy);border:2px solid #07222A;cursor:pointer;box-shadow:0 0 8px rgba(0,240,255,.5)}
.gc-slider input::-moz-range-thumb{width:12px;height:12px;border-radius:50%;background:var(--cy);border:2px solid #07222A;cursor:pointer}
.gc-collapse{display:flex;align-items:center;gap:8px;width:100%;background:none;border:0;color:var(--tx);font:inherit;font-size:11.5px;cursor:pointer;padding:0;text-align:left}
.gc-collapse svg:last-child{margin-left:auto;color:var(--dim)}
.gc-notes{margin-top:10px;font-size:11px;line-height:1.6;color:#9FB0C4}
.gc-notes h4{margin:10px 0 4px;font-size:11px;color:var(--am);font-weight:600}
.gc-notes p{margin:0 0 6px}
.gc-notes b{color:#D6E2EF}
@media (max-width:960px){
  .gc-root{overflow:auto;height:auto;min-height:100%}
  .gc-main{display:flex;flex-direction:column}
  .gc-stage{height:62vh;flex:none;border-right:0;border-bottom:1px solid var(--line)}
  .gc-actions{margin-left:0}
}
@media (prefers-reduced-motion:reduce){.gc-pill.cal i{animation:none}.gc-bar i,.gc-knob i{transition:none}}
`;
