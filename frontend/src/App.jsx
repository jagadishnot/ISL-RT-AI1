import { useState, useEffect, useRef, useCallback } from "react";

const WS_URL  = "ws://localhost:8000/ws/stream";
const API_URL = "http://localhost:8000/api";
const SIGN_IMG = (w) => `/signs/${w.toLowerCase().replace(/\s+/g, "_")}.jpg`;

// ── Browser TTS ───────────────────────────────────────────────────────────────
const speakText = (() => {
  let last = ""; let busy = false;
  return (text, muted = false) => {
    if (!text || muted || busy || text === last) return;
    if (!window.speechSynthesis) return;
    last = text; busy = true;
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 0.95; u.lang = "en-IN"; u.volume = 1.0;
    u.onend = () => { busy = false; };
    u.onerror = () => { busy = false; };
    window.speechSynthesis.speak(u);
  };
})();

// ── Tokens ────────────────────────────────────────────────────────────────────
const T = {
  bg: "#07090F", surf: "#0D1117", card: "#111827", border: "#1F2937",
  blue: "#3B82F6", cyan: "#06B6D4", teal: "#14B8A6", green: "#10B981",
  amber: "#F59E0B", red: "#EF4444", purple: "#8B5CF6",
  text: "#F1F5F9", sub: "#94A3B8", muted: "#475569",
};

// ── Global CSS ────────────────────────────────────────────────────────────────
const CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Fira+Code:wght@400;500&display=swap');
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { width: 100%; overflow-x: hidden; }
  body { background: #07090F; color: #F1F5F9; font-family: 'Space Grotesk', sans-serif; -webkit-font-smoothing: antialiased; }
  ::-webkit-scrollbar { width: 4px; } ::-webkit-scrollbar-track { background: #0D1117; } ::-webkit-scrollbar-thumb { background: #1F2937; border-radius: 4px; }
  @keyframes fadeUp { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
  @keyframes pulse  { 0%,100%{opacity:1} 50%{opacity:.4} }
  @keyframes spin   { to { transform:rotate(360deg); } }
  @keyframes ripple { 0%{transform:scale(1);opacity:.8} 100%{transform:scale(2.4);opacity:0} }
  @keyframes scan   { 0%{top:-5%} 100%{top:105%} }
  .fu { animation: fadeUp .3s ease both; }
  .mono { font-family:'Fira Code',monospace; }
  button:focus-visible { outline: 2px solid #06B6D4; outline-offset: 2px; }
  input:focus { outline: none; border-color: #06B6D4 !important; }
`;
function injectCSS() {
  if (document.getElementById("isl-css")) return;
  const s = document.createElement("style"); s.id = "isl-css"; s.textContent = CSS;
  document.head.appendChild(s);
}
function injectViewport() {
  if (document.querySelector('meta[name="viewport"]')) return;
  const m = document.createElement("meta"); m.name = "viewport";
  m.content = "width=device-width, initial-scale=1.0, maximum-scale=1.0";
  document.head.appendChild(m);
}

// ── Tiny components ───────────────────────────────────────────────────────────
function Dot({ on, color = T.green, label }) {
  return (
    <span style={{ display:"flex", alignItems:"center", gap:6, flexShrink:0 }}>
      <span style={{ position:"relative", width:8, height:8, display:"inline-block" }}>
        <span style={{ display:"block", width:8, height:8, borderRadius:"50%", background: on ? color : T.muted, transition:"background .3s" }}/>
        {on && <span style={{ position:"absolute", inset:0, borderRadius:"50%", background:color, animation:"ripple 1.4s ease-out infinite" }}/>}
      </span>
      {label && <span className="mono" style={{ fontSize:10, color: on ? color : T.muted, letterSpacing:.5, whiteSpace:"nowrap" }}>{label}</span>}
    </span>
  );
}

function Spinner({ size=20, color=T.cyan }) {
  return <span style={{ display:"inline-block", width:size, height:size, border:`2px solid ${color}33`, borderTop:`2px solid ${color}`, borderRadius:"50%", animation:"spin .75s linear infinite" }}/>;
}

function Tag({ children, color=T.cyan }) {
  return <span className="mono" style={{ fontSize:10, letterSpacing:1.5, padding:"3px 8px", borderRadius:4, background:`${color}18`, color, border:`1px solid ${color}30`, textTransform:"uppercase" }}>{children}</span>;
}

function Card({ children, style={}, glow }) {
  return (
    <div style={{ background:T.card, border:`1px solid ${glow ? glow+"44" : T.border}`, borderRadius:16, boxShadow: glow ? `0 0 24px ${glow}14` : "none", transition:"border-color .3s", ...style }}>
      {children}
    </div>
  );
}

function Btn({ children, onClick, disabled, variant="primary", style={} }) {
  const v = {
    primary: { background:`linear-gradient(135deg,${T.blue},${T.cyan})`, color:"#fff", border:"none" },
    danger:  { background:`${T.red}18`, color:T.red, border:`1px solid ${T.red}40` },
    ghost:   { background:"transparent", color:T.sub, border:`1px solid ${T.border}` },
  }[variant];
  return (
    <button onClick={onClick} disabled={disabled} style={{ padding:"10px 20px", borderRadius:10, fontSize:13, fontWeight:600, fontFamily:"'Space Grotesk',sans-serif", cursor:disabled?"not-allowed":"pointer", opacity:disabled?.5:1, transition:"all .2s", display:"flex", alignItems:"center", justifyContent:"center", gap:7, ...v, ...style }}>
      {children}
    </button>
  );
}

function PredBox({ text, signing }) {
  return (
    <Card glow={signing && text ? T.cyan : null} style={{ padding:"20px 24px", minHeight:76, display:"flex", alignItems:"center", justifyContent:"center" }}>
      {text
        ? <p key={text} className="fu" style={{ fontSize: text.length>40?18:text.length>20?24:30, fontWeight:700, textAlign:"center", background:`linear-gradient(90deg,${T.cyan},${T.teal},${T.blue})`, WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent" }}>{text}</p>
        : <p className="mono" style={{ fontSize:12, color:T.muted, animation: signing?"pulse 1.5s ease infinite":"none" }}>{signing?"reading sign…":"waiting for signs…"}</p>
      }
    </Card>
  );
}

function GText({ children }) {
  return <span style={{ background:`linear-gradient(90deg,${T.cyan},${T.blue})`, WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent" }}>{children}</span>;
}

// ── Stats Bar ─────────────────────────────────────────────────────────────────
function StatsBar() {
  const stats = [
    { label:"Word Accuracy", val:"95.4%",  color:T.cyan   },
    { label:"Exact Match",   val:"88.9%",  color:T.green  },
    { label:"WER",           val:"0.0458", color:T.teal   },
    { label:"Vocabulary",    val:"241",    color:T.purple  },
  ];
  return (
    <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:10, marginBottom:20 }}>
      {stats.map(s => (
        <Card key={s.label} style={{ padding:"12px 16px" }}>
          <p className="mono" style={{ fontSize:9, color:T.muted, letterSpacing:1.5, marginBottom:4 }}>{s.label.toUpperCase()}</p>
          <p style={{ fontSize:22, fontWeight:700, color:s.color }}>{s.val}</p>
        </Card>
      ))}
    </div>
  );
}

// ── TAB 1: Live Camera ────────────────────────────────────────────────────────
function LiveTab() {
  const videoRef  = useRef(null);
  const canvasRef = useRef(null);
  const wsRef     = useRef(null);
  const rafRef    = useRef(null);
  const fpsRef    = useRef({ count:0, last:Date.now() });
  const histRef   = useRef(null);

  const [camOn,   setCamOn]   = useState(false);
  const [wsState, setWsState] = useState("off");
  const [signing, setSigning] = useState(false);
  const [pred,    setPred]    = useState("");
  const [fps,     setFps]     = useState(0);
  const [history, setHistory] = useState([]);
  const [muted,   setMuted]   = useState(false);

  useEffect(() => { if (histRef.current) histRef.current.scrollTop = 0; }, [history.length]);

  const openWS = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    setWsState("connecting");
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;
    ws.onopen  = () => setWsState("on");
    ws.onclose = () => setWsState("off");
    ws.onerror = () => setWsState("error");
    ws.onmessage = ({ data }) => {
      const msg = JSON.parse(data);
      setSigning(msg.signing);
      if (msg.prediction) {
        setPred(p => {
          if (p !== msg.prediction) {
            setHistory(h => [{ text:msg.prediction, ts:new Date().toLocaleTimeString() }, ...h.slice(0,199)]);
            speakText(msg.prediction, muted);
          }
          return msg.prediction;
        });
      } else if (!msg.signing) { setPred(""); }
      fpsRef.current.count++;
      const now = Date.now();
      if (now - fpsRef.current.last >= 1000) { setFps(fpsRef.current.count); fpsRef.current = { count:0, last:now }; }
    };
  }, [muted]);

  const startLoop = useCallback(() => {
    const loop = () => {
      const video = videoRef.current, canvas = canvasRef.current, ws = wsRef.current;
      if (video && !video.paused && canvas) {
        canvas.width = video.videoWidth||640; canvas.height = video.videoHeight||480;
        canvas.getContext("2d").drawImage(video, 0, 0);
        if (ws?.readyState === WebSocket.OPEN) {
          canvas.toBlob(blob => {
            if (!blob) return;
            const r = new FileReader();
            r.onloadend = () => { if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ frame: r.result.split(",")[1] })); };
            r.readAsDataURL(blob);
          }, "image/jpeg", 0.55);
        }
      }
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
  }, []);

  const startCam = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video:{ width:640, height:480 } });
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setCamOn(true); openWS(); startLoop();
    } catch { alert("Camera access denied."); }
  }, [openWS, startLoop]);

  const stopCam = useCallback(() => {
    videoRef.current?.srcObject?.getTracks().forEach(t => t.stop());
    if (videoRef.current) videoRef.current.srcObject = null;
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    wsRef.current?.close();
    setCamOn(false); setSigning(false); setPred(""); setWsState("off");
  }, []);

  useEffect(() => () => stopCam(), [stopCam]);

  const wsColor = { off:T.muted, connecting:T.amber, on:T.green, error:T.red }[wsState];

  return (
    <div style={{ display:"grid", gridTemplateColumns:"1fr 280px", gap:16, width:"100%" }}>
      {/* Left: camera + prediction + controls */}
      <div style={{ display:"flex", flexDirection:"column", gap:12 }}>

        {/* Camera */}
        <Card style={{ position:"relative", overflow:"hidden", aspectRatio:"4/3", minHeight:240 }}>
          <video ref={videoRef} muted playsInline style={{ width:"100%", height:"100%", objectFit:"cover", display:camOn?"block":"none" }}/>
          <canvas ref={canvasRef} style={{ display:"none" }}/>
          {!camOn && (
            <div style={{ position:"absolute", inset:0, display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", gap:10, background:`repeating-linear-gradient(45deg,${T.surf} 0,${T.surf} 10px,${T.card} 10px,${T.card} 20px)` }}>
              <span style={{ fontSize:40, opacity:.2 }}>📷</span>
              <p className="mono" style={{ color:T.muted, fontSize:12 }}>camera offline</p>
            </div>
          )}
          {camOn && (
            <>
              <div style={{ position:"absolute", top:10, left:10, display:"flex", flexDirection:"column", gap:5, background:"#00000099", backdropFilter:"blur(8px)", borderRadius:8, padding:"7px 12px", border:`1px solid ${T.border}` }}>
                <Dot on={signing} color={T.cyan} label={signing?"SIGNING":"IDLE"}/>
                <Dot on={wsState==="on"} color={wsColor} label={wsState.toUpperCase()}/>
              </div>
              <div style={{ position:"absolute", top:10, right:10, background:"#00000099", backdropFilter:"blur(8px)", borderRadius:7, padding:"5px 10px", border:`1px solid ${T.border}` }}>
                <span className="mono" style={{ fontSize:11, color:T.green }}>{fps} fps</span>
              </div>
              {signing && <div style={{ position:"absolute", left:0, right:0, height:2, background:`linear-gradient(90deg,transparent,${T.cyan}88,transparent)`, animation:"scan 2s linear infinite", pointerEvents:"none" }}/>}
            </>
          )}
        </Card>

        {/* Prediction */}
        <PredBox text={pred} signing={signing}/>

        {/* Controls */}
        <div style={{ display:"flex", gap:8, flexWrap:"wrap" }}>
          {!camOn
            ? <Btn onClick={startCam} style={{ flex:1 }}>▶ Start Camera</Btn>
            : <Btn onClick={stopCam} variant="danger" style={{ flex:1 }}>■ Stop Camera</Btn>
          }
          <Btn onClick={() => { setMuted(m => !m); window.speechSynthesis?.cancel(); }} variant="ghost"
            style={{ minWidth:44, border:`1px solid ${muted?T.red+"60":T.border}`, color:muted?T.red:T.sub }}>
            {muted ? "🔇" : "🔊"}
          </Btn>
          {history.length > 0 && (
            <Btn onClick={() => setHistory([])} variant="ghost">Clear</Btn>
          )}
        </div>
      </div>

      {/* Right: history */}
      <Card style={{ padding:16, display:"flex", flexDirection:"column", gap:10, minHeight:200 }}>
        <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center" }}>
          <Tag color={T.cyan}>History</Tag>
          <span className="mono" style={{ fontSize:10, color:T.muted }}>{history.length}/200</span>
        </div>
        <div ref={histRef} style={{ flex:1, overflowY:"auto", display:"flex", flexDirection:"column", gap:6, maxHeight:440 }}>
          {history.length === 0
            ? <div style={{ textAlign:"center", marginTop:40 }}>
                <p style={{ fontSize:24, opacity:.2 }}>🤟</p>
                <p className="mono" style={{ color:T.muted, fontSize:11, marginTop:6 }}>no history yet</p>
              </div>
            : history.map((item, i) => (
                <div key={i} className={i===0?"fu":""} style={{ padding:"8px 11px", borderRadius:8, background:i===0?`${T.cyan}14`:T.surf, border:`1px solid ${i===0?T.cyan+"40":T.border}`, display:"flex", justifyContent:"space-between", alignItems:"center", gap:6 }}>
                  <span style={{ fontSize:13, fontWeight:600, color:i===0?T.text:T.sub, wordBreak:"break-word" }}>{item.text}</span>
                  <span className="mono" style={{ fontSize:9, color:T.muted, whiteSpace:"nowrap" }}>{item.ts}</span>
                </div>
              ))
          }
        </div>
      </Card>
    </div>
  );
}

// ── TAB 2: Upload Video ───────────────────────────────────────────────────────
function UploadTab() {
  const inputRef = useRef(null);
  const [file,    setFile]    = useState(null);
  const [preview, setPreview] = useState(null);
  const [drag,    setDrag]    = useState(false);
  const [loading, setLoading] = useState(false);
  const [result,  setResult]  = useState(null);
  const [err,     setErr]     = useState(null);

  const pick = (f) => {
    if (!f?.type.startsWith("video/")) { setErr("Please upload a video file."); return; }
    setFile(f); setResult(null); setErr(null);
    setPreview(URL.createObjectURL(f));
  };

  const translate = async () => {
    if (!file) return;
    setLoading(true); setResult(null); setErr(null);
    const fd = new FormData(); fd.append("file", file);
    try {
      const res  = await fetch(`${API_URL}/translate`, { method:"POST", body:fd });
      const data = await res.json();
      if (data.error) setErr(data.error);
      else { setResult(data); speakText(data.prediction || "", false); }
    } catch { setErr("Cannot reach backend. Make sure the server is running on port 8000."); }
    finally { setLoading(false); }
  };

  return (
    <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:16, alignItems:"start", width:"100%" }}>
      <div style={{ display:"flex", flexDirection:"column", gap:12 }}>
        {/* Drop zone */}
        <div
          onDragOver={e=>{e.preventDefault();setDrag(true);}} onDragLeave={()=>setDrag(false)}
          onDrop={e=>{e.preventDefault();setDrag(false);pick(e.dataTransfer.files[0]);}}
          onClick={()=>inputRef.current?.click()}
          style={{ border:`2px dashed ${drag?T.cyan:file?T.teal:T.border}`, borderRadius:16, minHeight:180, cursor:"pointer", display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", gap:10, background:drag?`${T.cyan}0A`:T.card, transition:"all .3s", padding:24 }}>
          <input ref={inputRef} type="file" accept="video/*" style={{ display:"none" }} onChange={e=>pick(e.target.files[0])}/>
          <span style={{ fontSize:38, opacity:file?1:.25 }}>{file?"🎬":"📂"}</span>
          {file
            ? <div style={{ textAlign:"center" }}>
                <p style={{ fontWeight:700, fontSize:14 }}>{file.name}</p>
                <p className="mono" style={{ fontSize:11, color:T.muted, marginTop:3 }}>{(file.size/1024/1024).toFixed(2)} MB</p>
              </div>
            : <div style={{ textAlign:"center" }}>
                <p style={{ fontWeight:600, color:T.sub, fontSize:14 }}>Drop sign language video here</p>
                <p className="mono" style={{ fontSize:11, color:T.muted, marginTop:3 }}>mp4 · avi · mov · webm</p>
              </div>
          }
        </div>
        {preview && <video src={preview} controls style={{ width:"100%", borderRadius:12, border:`1px solid ${T.border}`, background:"#000" }}/>}
        <Btn onClick={translate} disabled={!file||loading} style={{ width:"100%" }}>
          {loading ? <><Spinner size={15}/> Translating…</> : "→ Translate to Text"}
        </Btn>
      </div>

      {/* Result */}
      <Card style={{ padding:24, minHeight:280, display:"flex", flexDirection:"column", justifyContent:"center" }}>
        {!result && !err && !loading && (
          <div style={{ textAlign:"center" }}>
            <span style={{ fontSize:40, opacity:.12 }}>💬</span>
            <p className="mono" style={{ color:T.muted, fontSize:12, marginTop:10 }}>translation will appear here</p>
          </div>
        )}
        {loading && (
          <div style={{ display:"flex", flexDirection:"column", alignItems:"center", gap:16 }}>
            <Spinner size={40} color={T.teal}/>
            <p style={{ fontWeight:600, color:T.sub, fontSize:14 }}>Processing video…</p>
            <p className="mono" style={{ fontSize:11, color:T.muted }}>extracting landmarks & running inference</p>
          </div>
        )}
        {err && !loading && (
          <div className="fu" style={{ padding:14, borderRadius:10, background:`${T.red}14`, border:`1px solid ${T.red}40` }}>
            <p style={{ color:T.red, fontSize:13 }}>⚠ {err}</p>
          </div>
        )}
        {result && !loading && (
          <div className="fu" style={{ display:"flex", flexDirection:"column", gap:16 }}>
            <div style={{ padding:"18px 20px", borderRadius:12, background:`linear-gradient(135deg,${T.blue}20,${T.cyan}0A)`, border:`1px solid ${T.cyan}30` }}>
              <Tag color={T.cyan}>Prediction</Tag>
              <p style={{ fontSize: result.prediction?.length>30?20:26, fontWeight:700, marginTop:8, background:`linear-gradient(90deg,${T.cyan},${T.teal})`, WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent" }}>
                {result.prediction || "(no sign detected)"}
              </p>
            </div>
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:8 }}>
              {[["Frames", result.frames], ["Confidence", result.confidence ? `${(result.confidence*100).toFixed(0)}%` : "—"]].map(([l,v]) => (
                <div key={l} style={{ padding:"10px 14px", borderRadius:9, background:T.surf, border:`1px solid ${T.border}` }}>
                  <p className="mono" style={{ fontSize:9, color:T.muted, letterSpacing:1.5, marginBottom:4 }}>{l.toUpperCase()}</p>
                  <p style={{ fontSize:20, fontWeight:700 }}>{v}</p>
                </div>
              ))}
            </div>
            <Btn onClick={() => speakText(result.prediction || "", false)} style={{ width:"100%" }} variant="ghost">🔊 Speak Again</Btn>
          </div>
        )}
      </Card>
    </div>
  );
}

// ── TAB 3: Text → Sign ────────────────────────────────────────────────────────
function SignCard({ word }) {
  const [imgOk, setImgOk] = useState(true);
  return (
    <div className="fu" style={{ borderRadius:12, overflow:"hidden", border:`1px solid ${T.border}`, background:T.card }}>
      <div style={{ aspectRatio:"1", background:T.surf, display:"flex", alignItems:"center", justifyContent:"center", overflow:"hidden" }}>
        {imgOk
          ? <img src={SIGN_IMG(word)} alt={word} onError={()=>setImgOk(false)} style={{ width:"100%", height:"100%", objectFit:"cover" }}/>
          : <div style={{ display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", gap:6, width:"100%", height:"100%", background:`linear-gradient(135deg,${T.blue}18,${T.purple}18)` }}>
              <span style={{ fontSize:28 }}>🤟</span>
              <span className="mono" style={{ fontSize:9, color:T.muted }}>no image</span>
            </div>
        }
      </div>
      <div style={{ padding:"6px 8px", textAlign:"center" }}>
        <p style={{ fontSize:12, fontWeight:600, color:T.sub }}>{word}</p>
      </div>
    </div>
  );
}

function TextToSignTab({ vocab }) {
  const [input,   setInput]   = useState("");
  const [words,   setWords]   = useState([]);
  const [curr,    setCurr]    = useState(-1);
  const [playing, setPlaying] = useState(false);
  const playRef = useRef(null);

  const EXAMPLES = ["hello thank you", "bear monkey uncle", "volcano wrong wife"];

  const submit = (txt) => {
    const raw = (txt || input).trim().toLowerCase();
    if (!raw) return;
    const ws = raw.split(/\s+/).filter(Boolean);
    setWords(ws); setCurr(-1); setPlaying(false);
    speakText(raw, false);
  };

  const play = () => {
    if (!words.length) return;
    setPlaying(true); setCurr(0); speakText(words[0], false);
    let i = 0;
    playRef.current = setInterval(() => {
      i++;
      if (i >= words.length) { clearInterval(playRef.current); setPlaying(false); setCurr(-1); }
      else { setCurr(i); speakText(words[i], false); }
    }, 1600);
  };

  const stopPlay = () => { clearInterval(playRef.current); setPlaying(false); setCurr(-1); };
  useEffect(() => () => clearInterval(playRef.current), []);
  const inVocab = (w) => vocab.some(v => v.toLowerCase() === w.toLowerCase());

  return (
    <div style={{ display:"flex", flexDirection:"column", gap:16 }}>
      <Card style={{ padding:20 }}>
        <div style={{ display:"flex", flexDirection:"column", gap:12 }}>
          <div style={{ display:"flex", gap:10 }}>
            <input value={input} onChange={e=>setInput(e.target.value)} onKeyDown={e=>e.key==="Enter"&&submit()}
              placeholder="Type words or a sentence…"
              style={{ flex:1, padding:"10px 14px", background:T.surf, border:`1px solid ${T.border}`, borderRadius:10, color:T.text, fontSize:15, fontFamily:"'Space Grotesk',sans-serif", transition:"border-color .2s" }}/>
            <Btn onClick={()=>submit()} disabled={!input.trim()}>Show Signs</Btn>
          </div>
          <div style={{ display:"flex", gap:8, flexWrap:"wrap", alignItems:"center" }}>
            <span className="mono" style={{ fontSize:10, color:T.muted }}>Try:</span>
            {EXAMPLES.map(ex => (
              <button key={ex} onClick={()=>{setInput(ex);submit(ex);}}
                style={{ background:T.surf, border:`1px solid ${T.border}`, borderRadius:7, padding:"4px 11px", color:T.sub, fontSize:12, cursor:"pointer", fontFamily:"'Space Grotesk',sans-serif" }}>
                {ex}
              </button>
            ))}
          </div>
        </div>
      </Card>

      {words.length > 0 && (
        <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", flexWrap:"wrap", gap:8 }}>
          <div style={{ display:"flex", gap:10, alignItems:"center" }}>
            <Tag color={T.teal}>{words.length} word{words.length>1?"s":""}</Tag>
            <Tag color={words.filter(w=>!inVocab(w)).length===0?T.green:T.amber}>
              {words.filter(w=>!inVocab(w)).length===0?"all in vocabulary":`${words.filter(w=>!inVocab(w)).length} unknown`}
            </Tag>
          </div>
          <div style={{ display:"flex", gap:8 }}>
            {!playing
              ? <Btn onClick={play} style={{ padding:"8px 16px", fontSize:12 }}>▶ Auto-play</Btn>
              : <Btn onClick={stopPlay} variant="danger" style={{ padding:"8px 16px", fontSize:12 }}>■ Stop</Btn>
            }
            <Btn onClick={()=>speakText(words.join(" "), false)} variant="ghost" style={{ padding:"8px 14px", fontSize:12 }}>🔊</Btn>
            <Btn onClick={()=>{setWords([]);setInput("");setCurr(-1);stopPlay();}} variant="ghost" style={{ padding:"8px 14px", fontSize:12 }}>Clear</Btn>
          </div>
        </div>
      )}

      {words.length > 0 && (
        <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fill,minmax(120px,1fr))", gap:12 }}>
          {words.map((word, i) => (
            <div key={i} style={{ outline: curr===i?`2px solid ${T.cyan}`:"2px solid transparent", outlineOffset:3, borderRadius:14, transform:curr===i?"scale(1.04)":"scale(1)", transition:"outline .2s, transform .2s" }}>
              <SignCard word={word}/>
              {!inVocab(word) && <p className="mono" style={{ fontSize:9, color:T.amber, textAlign:"center", marginTop:3 }}>not in vocab</p>}
            </div>
          ))}
        </div>
      )}

      {words.length === 0 && (
        <Card style={{ padding:48, textAlign:"center" }}>
          <span style={{ fontSize:44, opacity:.12 }}>🤲</span>
          <p className="mono" style={{ color:T.muted, marginTop:10, fontSize:12 }}>enter text above to see sign language cards</p>
          <p style={{ color:T.muted, fontSize:12, marginTop:5 }}>Add images to <code style={{ color:T.cyan }}>public/signs/word.jpg</code></p>
        </Card>
      )}
    </div>
  );
}

// ── ROOT ──────────────────────────────────────────────────────────────────────
export default function App() {
  injectCSS();
  injectViewport();

  const [tab,   setTab]   = useState("live");
  const [apiOk, setApiOk] = useState(null);
  const [vocab, setVocab] = useState([]);

  useEffect(() => {
    fetch(`${API_URL.replace("/api","")}/health`).then(r=>r.json()).then(()=>setApiOk(true)).catch(()=>setApiOk(false));
    fetch(`${API_URL}/vocab`).then(r=>r.json()).then(d=>setVocab(d.vocab||[])).catch(()=>{});
  }, []);

  const TABS = [
    { id:"live",   icon:"📷", label:"Live Camera"  },
    { id:"upload", icon:"📁", label:"Upload Video"  },
    { id:"text",   icon:"✍️",  label:"Text → Sign"  },
  ];

  return (
    <div style={{ minHeight:"100vh", background:T.bg }}>

      {/* Header */}
      <header style={{ borderBottom:`1px solid ${T.border}`, background:`${T.surf}F0`, backdropFilter:"blur(16px)", position:"sticky", top:0, zIndex:100 }}>
        <div style={{ width:"100%", padding:"0 20px", height:56, display:"flex", alignItems:"center", justifyContent:"space-between", gap:12 }}>

          {/* Brand */}
          <div style={{ display:"flex", alignItems:"center", gap:10, flexShrink:0 }}>
            <div style={{ width:34, height:34, borderRadius:9, background:`linear-gradient(135deg,${T.blue},${T.cyan})`, display:"flex", alignItems:"center", justifyContent:"center", fontSize:18 }}>🤟</div>
            <div>
              <p style={{ fontSize:15, fontWeight:700, letterSpacing:"-0.3px", lineHeight:1.1 }}>ISL-RT-AI1</p>
              <p className="mono" style={{ fontSize:9, color:T.muted, letterSpacing:2 }}>SIGN LANGUAGE AI</p>
            </div>
          </div>

          {/* Tabs */}
          <nav style={{ display:"flex", gap:3, background:T.card, borderRadius:10, padding:3, border:`1px solid ${T.border}` }}>
            {TABS.map(t => (
              <button key={t.id} onClick={()=>setTab(t.id)} style={{
                display:"flex", alignItems:"center", gap:5, padding:"7px 14px", borderRadius:7, border:"none", cursor:"pointer",
                background: tab===t.id ? `linear-gradient(135deg,${T.blue}CC,${T.cyan}88)` : "transparent",
                color: tab===t.id ? T.text : T.muted, fontSize:13, fontWeight:600,
                fontFamily:"'Space Grotesk',sans-serif", transition:"all .18s", whiteSpace:"nowrap",
              }}>
                <span style={{ fontSize:14 }}>{t.icon}</span>
                <span>{t.label}</span>
              </button>
            ))}
          </nav>

          {/* API status */}
          <Dot on={apiOk===true} color={T.green} label={apiOk===null?"CHECKING…":apiOk?"API ONLINE":"API OFFLINE"}/>
        </div>
      </header>

      {/* Main */}
      <main style={{ width:"100%", padding:"20px 20px 60px" }}>

        {/* Page title */}
        <div style={{ marginBottom:18 }}>
          <h1 style={{ fontSize:30, fontWeight:700, letterSpacing:"-0.5px", lineHeight:1.2 }}>
            {tab==="live"   && <>Live <GText>Sign Translator</GText></>}
            {tab==="upload" && <>Video <GText>Translation</GText></>}
            {tab==="text"   && <>Text to <GText>Sign Language</GText></>}
          </h1>
          <p style={{ color:T.muted, marginTop:6, fontSize:13 }}>
            {tab==="live"   && "Real-time ISL recognition via webcam — GNN + Transformer model"}
            {tab==="upload" && "Upload a sign language video and get the text translation"}
            {tab==="text"   && "Enter text to see the corresponding ISL signs"}
          </p>
        </div>

        <StatsBar/>

        {tab==="live"   && <LiveTab/>}
        {tab==="upload" && <UploadTab/>}
        {tab==="text"   && <TextToSignTab vocab={vocab}/>}
      </main>
    </div>
  );
}