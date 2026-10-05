import { useState, useRef, useEffect } from 'react'
import { Upload, Activity, ShieldCheck, Database, BarChart3, AlertCircle, RefreshCw, Train, ExternalLink, Settings, Terminal } from 'lucide-react'

// Configuration
const API_URL = "http://localhost:8000"; 
const DAGSHUB_URL = "https://dagshub.com/Abhishek-B16/railway-defect-detection-mlops";
const MLFLOW_UI_URL = "https://dagshub.com/Abhishek-B16/railway-defect-detection-mlops.mlflow";
const GRAFANA_URL = "http://localhost:3000";

const COLORS = {
  crack: "#ef4444", spalling: "#f97316", flaking: "#eab308",
  bolts: "#06b6d4", joints: "#3b82f6", sheling: "#d946ef"
};

function App() {
  const [activeTab, setActiveTab] = useState("diagnostics");
  const [file, setFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [detections, setDetections] = useState([]);
  const [error, setError] = useState(null);
  const [apiStatus, setApiStatus] = useState("checking");
  const [modelInfo, setModelInfo] = useState(null);
  
  const canvasRef = useRef(null);
  const imgRef = useRef(null);

  useEffect(() => { checkHealth(); }, []);

  const checkHealth = async () => {
    setApiStatus("checking");
    try {
      const res = await fetch(`${API_URL}/health`);
      if (res.ok) {
        setApiStatus("online");
        fetchModelInfo();
      } else {
        setApiStatus("offline");
      }
    } catch (e) { setApiStatus("offline"); }
  };

  const fetchModelInfo = async () => {
    try {
      const res = await fetch(`${API_URL}/model-info`);
      if (res.ok) {
        const data = await res.json();
        setModelInfo(data);
      }
    } catch (e) { console.error(e); }
  };

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
      setImagePreview(URL.createObjectURL(selected));
      setDetections([]);
      setError(null);
      if (canvasRef.current) {
        const ctx = canvasRef.current.getContext('2d');
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
    }
  };

  const handleDetect = async () => {
    if (!file) return;
    setLoading(true); setError(null); setDetections([]);
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await fetch(`${API_URL}/predict`, { method: 'POST', body: formData });
      if (!res.ok) throw new Error(`API Error: ${res.status}`);
      const data = await res.json();
      setDetections(data.detections || []);
      drawDetections(data.detections || []);
    } catch (err) { setError(err.message); } 
    finally { setLoading(false); }
  };

  const drawDetections = (dets) => {
    const img = imgRef.current;
    const canvas = canvasRef.current;
    if (!img || !canvas) return;
    canvas.width = img.clientWidth;
    canvas.height = img.clientHeight;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const scaleX = img.clientWidth / img.naturalWidth;
    const scaleY = img.clientHeight / img.naturalHeight;

    dets.forEach(det => {
      const { x1, y1, x2, y2 } = det.bbox;
      const color = COLORS[det.class_name] || "#ef4444";
      const sx1 = x1 * scaleX, sy1 = y1 * scaleY, sw = (x2 - x1) * scaleX, sh = (y2 - y1) * scaleY;
      ctx.strokeStyle = color; ctx.lineWidth = 3; ctx.strokeRect(sx1, sy1, sw, sh);
      const text = `${det.class_name} ${(det.confidence * 100).toFixed(1)}%`;
      ctx.font = "bold 13px Inter";
      const tw = ctx.measureText(text).width;
      ctx.fillStyle = color; ctx.fillRect(sx1, sy1 - 22, tw + 10, 22);
      ctx.fillStyle = "#000"; ctx.fillText(text, sx1 + 5, sy1 - 6);
    });
  };

  return (
    <div className="app-container">
      <div className="sidebar">
        <div className="sidebar-header">
          <Train size={24} color="var(--accent)" />
          <h1 style={{ fontWeight: 700, letterSpacing: '-0.5px' }}>RailGuard</h1>
        </div>

        <div className="sidebar-menu">
          <div style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: 'var(--text-secondary)', padding: '10px 16px', letterSpacing: '1px', fontWeight: 600 }}>Tools</div>
          
          <button className={`menu-item ${activeTab === 'diagnostics' ? 'active' : ''}`} onClick={() => setActiveTab('diagnostics')}>
            <Activity size={18} /> Vision Diagnostics
          </button>
          
          <a href={`${API_URL}/docs`} target="_blank" rel="noreferrer" className="menu-item">
            <Terminal size={18} /> API Console <ExternalLink size={14} style={{marginLeft: 'auto', opacity: 0.5}}/>
          </a>
          
          <a href={DAGSHUB_URL} target="_blank" rel="noreferrer" className="menu-item">
            <Database size={18} /> DagsHub Registry <ExternalLink size={14} style={{marginLeft: 'auto', opacity: 0.5}}/>
          </a>
          
          <a href={MLFLOW_UI_URL} target="_blank" rel="noreferrer" className="menu-item">
            <Settings size={18} /> MLflow Tracking <ExternalLink size={14} style={{marginLeft: 'auto', opacity: 0.5}}/>
          </a>

          <a href={GRAFANA_URL} target="_blank" rel="noreferrer" className="menu-item">
            <BarChart3 size={18} /> Grafana Metrics <ExternalLink size={14} style={{marginLeft: 'auto', opacity: 0.5}}/>
          </a>
        </div>

        <div className="sidebar-status">
          <div className="status-badge" style={{ marginBottom: '16px' }}>
            <div className={`status-dot ${apiStatus === 'online' ? 'online' : ''}`} style={{boxShadow: `0 0 12px ${apiStatus === 'online' ? 'var(--success)' : 'var(--error)'}`}}></div>
            <span style={{fontWeight: 500, letterSpacing: '0.5px'}}>{apiStatus === 'online' ? 'FastAPI Connected' : 'API Disconnected'}</span>
          </div>
          
          {modelInfo && (
            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', background: 'var(--bg-tertiary)', padding: '12px', borderRadius: '8px', border: '1px solid var(--border)' }}>
              <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: '4px'}}>
                <span>Model:</span> <span style={{color: 'var(--text-primary)'}}>{modelInfo.model_name}</span>
              </div>
              <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: '4px'}}>
                <span>Version:</span> <span style={{color: 'var(--accent)'}}>v{modelInfo.version}</span>
              </div>
              <div style={{display: 'flex', justifyContent: 'space-between'}}>
                <span>Alias:</span> <span style={{color: 'var(--success)'}}>@{modelInfo.alias}</span>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="main-content">
        {activeTab === 'diagnostics' && (
          <>
            <div className="header" style={{textAlign: 'left', marginBottom: '50px'}}>
              <h2 style={{ fontSize: '3rem', letterSpacing: '-1px', margin: 0 }}>Vision Diagnostics</h2>
              <p style={{ color: 'var(--text-secondary)', fontSize: '1.1rem', marginTop: '8px' }}>Upload railway infrastructure imagery for automated anomaly detection using YOLOv8.</p>
            </div>

            {!file ? (
              <label className="upload-area" style={{ padding: '80px 20px', borderStyle: 'dashed' }}>
                <Upload size={54} className="upload-icon" style={{ marginBottom: '24px' }} />
                <h3 style={{ fontSize: '1.5rem', fontWeight: 600 }}>Drag & Drop Image Here</h3>
                <p style={{ marginTop: '12px', color: 'var(--text-secondary)', fontSize: '1rem' }}>Supports JPG, PNG up to 10MB</p>
                <input type="file" className="hidden-input" accept="image/jpeg, image/png, image/jpg" onChange={handleFileChange} />
              </label>
            ) : (
              <div className="analysis-layout">
                <div className="image-panel">
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                    <h3 style={{fontSize: '1rem', textTransform: 'uppercase', letterSpacing: '1px', color: 'var(--text-secondary)'}}>Source Feed</h3>
                    <button className="btn-secondary" style={{fontSize: '0.8rem', padding: '6px 12px'}} onClick={() => { setFile(null); setImagePreview(null); }}>
                      Clear
                    </button>
                  </div>
                  
                  <div className="canvas-container" style={{ border: '1px solid var(--border)', borderRadius: '12px' }}>
                    <img ref={imgRef} src={imagePreview} alt="Track" onLoad={() => { if (detections.length > 0) drawDetections(detections); }} />
                    <canvas ref={canvasRef} />
                  </div>
                </div>

                <div className="results-panel">
                  <h3 style={{fontSize: '1rem', textTransform: 'uppercase', letterSpacing: '1px', color: 'var(--text-secondary)', marginBottom: '20px'}}>Analysis Engine</h3>

                  <button className="btn-primary" onClick={handleDetect} disabled={loading || apiStatus !== 'online'} style={{ padding: '16px', fontSize: '1.1rem', borderRadius: '12px' }}>
                    {loading ? (
                      <><RefreshCw size={20} className="spin" /> Processing Neural Network...</>
                    ) : (
                      <><ShieldCheck size={20} /> Run Defect Analysis</>
                    )}
                  </button>

                  {error && (
                    <div style={{ marginTop: '24px', padding: '16px', background: 'rgba(239, 68, 68, 0.1)', color: 'var(--error)', borderRadius: '12px', border: '1px solid rgba(239, 68, 68, 0.2)', display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
                      <AlertCircle size={20} style={{flexShrink: 0, marginTop: '2px'}} /> 
                      <span style={{fontSize: '0.95rem', lineHeight: 1.5}}>{error}</span>
                    </div>
                  )}

                  {detections.length > 0 && (
                    <div className="detections-list" style={{marginTop: '30px'}}>
                      <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px'}}>
                        <h4 style={{ color: 'var(--text-primary)', fontSize: '1.2rem', fontWeight: 600 }}>Anomalies Detected</h4>
                        <span style={{ background: 'var(--error)', color: '#fff', padding: '4px 12px', borderRadius: '20px', fontSize: '0.85rem', fontWeight: 700 }}>{detections.length} Found</span>
                      </div>
                      
                      <div style={{display: 'flex', flexDirection: 'column', gap: '12px'}}>
                        {detections.map((det, i) => (
                          <div className="detection-item" key={i} style={{ borderLeftColor: COLORS[det.class_name] || 'red', background: 'var(--bg-tertiary)', padding: '16px', borderRadius: '12px' }}>
                            <div style={{display: 'flex', alignItems: 'center', gap: '12px'}}>
                              <div style={{width: '12px', height: '12px', borderRadius: '50%', background: COLORS[det.class_name] || 'red'}}></div>
                              <span className="det-label" style={{fontSize: '1.1rem'}}>{det.class_name}</span>
                            </div>
                            <span className="det-conf" style={{background: 'var(--bg-secondary)', padding: '6px 12px', borderRadius: '6px', color: 'var(--text-secondary)'}}>
                              {(det.confidence * 100).toFixed(1)}% Match
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {!loading && detections.length === 0 && !error && file && apiStatus === 'online' && (
                    <div className="empty-state" style={{marginTop: '40px'}}>
                      <Activity size={48} style={{opacity: 0.2, marginBottom: '20px'}} />
                      <h4 style={{fontSize: '1.2rem', marginBottom: '8px', color: 'var(--text-primary)'}}>System Standby</h4>
                      <p style={{fontSize: '0.95rem'}}>Neural network is loaded and ready for inference.</p>
                    </div>
                  )}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}

export default App
