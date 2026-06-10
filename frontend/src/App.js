import React, { useState, useRef, useEffect, useCallback } from 'react';
import axios from 'axios';
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import './index.css';

const API = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';
const COLORS = ['#58a6ff','#3fb950','#f0883e','#f85149','#d2a8ff','#79c0ff'];

// ── Example queries ───────────────────────────────────────────────────────────
const EXAMPLES = [
  'Which subject has the most failures across Nilgiris?',
  'Compare pass % between boys and girls in Gudalur',
  'Show a bar chart of school-wise Math averages',
  'Top 5 schools by English average score',
  'How many students failed overall in 2026?',
  'Failures in Physics vs Chemistry comparison',
  'What is the re-examination policy?',
  'Trend of pass rates across 2025 and 2026',
];

// ── Utility components ────────────────────────────────────────────────────────
function Badge({ type, children }) {
  return <span className={`badge badge-${type}`}>{children}</span>;
}

function TypeBadge({ type }) {
  return <Badge type={type}>{type.toUpperCase()}</Badge>;
}

function JudgeBadge({ score }) {
  if (!score) return null;
  const level = score >= 4 ? 'high' : score === 3 ? 'medium' : 'low';
  return <Badge type={`score-${score}`}>Judge {score}/5</Badge>;
}

function LoadingDots() {
  return (
    <div className="loading-dots">
      <div className="dot" /><div className="dot" /><div className="dot" />
    </div>
  );
}

// ── Chart renderer ────────────────────────────────────────────────────────────
function ChartPanel({ chart }) {
  if (!chart || !chart.chart_json?.data?.length) return null;
  const plotData = chart.chart_json.data[0];
  const xVals = plotData.x || plotData.labels || [];
  const yVals = plotData.y || plotData.values || [];
  const data  = xVals.map((x, i) => ({ name: String(x), value: Number(yVals[i]) || 0 }));

  return (
    <div className="chart-card">
      <div className="chart-title">{chart.title}</div>
      <ResponsiveContainer width="100%" height={220}>
        {chart.chart_type === 'pie' ? (
          <PieChart>
            <Pie data={data} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label>
              {data.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip contentStyle={{ background: '#161b22', border: '1px solid #21262d', borderRadius: 8, fontSize: 11 }} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
          </PieChart>
        ) : chart.chart_type === 'line' ? (
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
            <XAxis dataKey="name" stroke="#6e7681" tick={{ fontSize: 10 }} />
            <YAxis stroke="#6e7681" tick={{ fontSize: 10 }} />
            <Tooltip contentStyle={{ background: '#161b22', border: '1px solid #21262d', borderRadius: 8, fontSize: 11 }} />
            <Line type="monotone" dataKey="value" stroke="#58a6ff" strokeWidth={2} dot={{ fill: '#58a6ff', r: 3 }} />
          </LineChart>
        ) : (
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
            <XAxis dataKey="name" stroke="#6e7681" tick={{ fontSize: 9 }} angle={-20} textAnchor="end" height={45} />
            <YAxis stroke="#6e7681" tick={{ fontSize: 10 }} />
            <Tooltip contentStyle={{ background: '#161b22', border: '1px solid #21262d', borderRadius: 8, fontSize: 11 }} />
            <Bar dataKey="value" radius={[4, 4, 0, 0]}>
              {data.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Bar>
          </BarChart>
        )}
      </ResponsiveContainer>
    </div>
  );
}

// ── Recommendations ───────────────────────────────────────────────────────────
function Recommendations({ recs }) {
  if (!recs?.length) return null;
  const priorities = ['high', 'medium', 'low'];
  return (
    <div className="recs">
      <div style={{ fontSize: 11, color: '#6e7681', marginBottom: 4, fontWeight: 600, textTransform: 'uppercase', letterSpacing: 1 }}>
        Recommended Actions
      </div>
      {recs.slice(0, 3).map((r, i) => (
        <div key={i} className={`rec-card ${priorities[i] || 'low'}`}>
          <div className="rec-header">
            <span className="rec-rank">#{r.rank || i + 1}</span>
            <span style={{ fontSize: 10, color: '#6e7681', textTransform: 'capitalize' }}>
              {priorities[i] || 'low'} priority
            </span>
          </div>
          <div className="rec-text">{r.text || r}</div>
        </div>
      ))}
    </div>
  );
}

// ── Metrics panel ─────────────────────────────────────────────────────────────
function MetricsPanel({ metrics }) {
  if (!metrics) return null;
  const breakdown = metrics.query_type_breakdown || {};
  const total = metrics.total_requests || 1;
  return (
    <div className="metrics-panel">
      <div style={{ fontSize: 11, fontWeight: 600, color: '#8b949e' }}>System Metrics</div>
      <div className="metrics-grid">
        <div className="metric-box">
          <div className="metric-val">{metrics.total_requests || 0}</div>
          <div className="metric-lbl">Total Queries</div>
        </div>
        <div className="metric-box">
          <div className="metric-val">{metrics.avg_latency_ms || 0}ms</div>
          <div className="metric-lbl">Avg Latency</div>
        </div>
        <div className="metric-box">
          <div className="metric-val">{metrics.avg_judge_score || '—'}</div>
          <div className="metric-lbl">Avg Judge Score</div>
        </div>
        <div className="metric-box">
          <div className="metric-val">${(metrics.total_cost_usd || 0).toFixed(4)}</div>
          <div className="metric-lbl">Total Cost</div>
        </div>
      </div>
      {Object.keys(breakdown).length > 0 && (
        <div className="routing-bar">
          {Object.entries(breakdown).map(([type, count]) => (
            <div key={type} className="routing-item">
              <div className="routing-label">
                <span style={{ textTransform: 'capitalize' }}>{type}</span>
                <span>{count}</span>
              </div>
              <div className="routing-track">
                <div className="routing-fill" style={{ width: `${(count / total) * 100}%` }} />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Message component ─────────────────────────────────────────────────────────
function Message({ msg }) {
  const [showSql, setShowSql] = useState(false);

  if (msg.role === 'user') {
    return (
      <div className="msg">
        <div className="msg-user">
          <div className="msg-user-bubble">{msg.content}</div>
        </div>
      </div>
    );
  }

  const r = msg.response;
  if (!r) return null;

  return (
    <div className="msg">
      <div className="msg-assistant">
        <div className="msg-avatar">D</div>
        <div className="msg-content">
          <div className="msg-header">
            <span className="msg-label">DMU Analytics</span>
            {r.query_type && <TypeBadge type={r.query_type} />}
            <JudgeBadge score={r.judge_score} />
            <span style={{ fontSize: 10, color: '#6e7681', marginLeft: 'auto' }}>
              {r.latency_ms}ms · {r.tokens} tokens · ${(r.cost_usd || 0).toFixed(5)}
            </span>
          </div>

          <div className="answer-card">
            <div style={{ whiteSpace: 'pre-wrap', lineHeight: 1.6 }}>
              {r.answer}
            </div>

            {r.error && (
              <div style={{ marginTop: 8, fontSize: 11, color: '#f85149' }}>
                Error: {r.error}
              </div>
            )}

            {r.sql && (
              <div>
                <button className="sql-toggle" onClick={() => setShowSql(v => !v)}>
                  {showSql ? '▼ Hide SQL' : '▶ Show generated SQL'}
                </button>
                {showSql && <pre className="sql-block">{r.sql}</pre>}
              </div>
            )}

            {r.judge_issues?.length > 0 && (
              <div style={{ marginTop: 6, fontSize: 11, color: '#e3b341' }}>
                ⚠ {r.judge_issues.join(' · ')}
              </div>
            )}

            <div className="answer-meta">
              <span style={{ fontSize: 10, color: '#6e7681' }}>
                {r.attempts > 1 ? `${r.attempts} attempts` : '1 attempt'}
              </span>
            </div>
          </div>

          {r.chart && <ChartPanel chart={r.chart} />}
          {r.recommendations?.length > 0 && <Recommendations recs={r.recommendations} />}
        </div>
      </div>
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [messages,  setMessages]  = useState([]);
  const [input,     setInput]     = useState('');
  const [loading,   setLoading]   = useState(false);
  const [health,    setHealth]    = useState(null);
  const [metrics,   setMetrics]   = useState(null);
  const [showMetrics, setShowMetrics] = useState(false);
  const textareaRef = useRef(null);
  const bottomRef   = useRef(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height =
        Math.min(textareaRef.current.scrollHeight, 120) + 'px';
    }
  }, [input]);

  // Scroll to bottom on new message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Load health + metrics
  useEffect(() => {
    const load = async () => {
      try {
        const [h, m] = await Promise.all([
          axios.get(`${API}/health`),
          axios.get(`${API}/metrics`),
        ]);
        setHealth(h.data);
        setMetrics(m.data);
      } catch {}
    };
    load();
    const interval = setInterval(load, 30000);
    return () => clearInterval(interval);
  }, []);

  const send = useCallback(async () => {
    const q = input.trim();
    if (!q || loading) return;

    setMessages(prev => [...prev, { role: 'user', content: q }]);
    setInput('');
    setLoading(true);

    try {
      const { data } = await axios.post(`${API}/query`, {
        query: q,
        show_sql: true,
        show_judge: true,
      });
      setMessages(prev => [...prev, { role: 'assistant', response: data }]);
      // Refresh metrics
      const m = await axios.get(`${API}/metrics`);
      setMetrics(m.data);
    } catch (e) {
      // Safely stringify FastAPI error detail (can be string or array of validation objects)
      const rawDetail = e.response?.data?.detail;
      let errorMsg;
      if (!rawDetail) {
        errorMsg = e.message || 'Unknown error';
      } else if (typeof rawDetail === 'string') {
        errorMsg = rawDetail;
      } else if (Array.isArray(rawDetail)) {
        errorMsg = rawDetail.map(d => `${d.loc?.join('.')} — ${d.msg}`).join('; ');
      } else {
        errorMsg = JSON.stringify(rawDetail);
      }
      setMessages(prev => [...prev, {
        role: 'assistant',
        response: {
          query: q,
          query_type: 'error',
          answer: `Error: ${errorMsg}`,
          latency_ms: 0, tokens: 0, cost_usd: 0, attempts: 1,
        }
      }]);
    } finally {
      setLoading(false);
    }
  }, [input, loading]);

  const onKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  };

  const history = messages.filter(m => m.role === 'user');

  return (
    <div className="app">
      {/* Sidebar */}
      <div className="sidebar">
        <div className="sidebar-header">
          <div className="sidebar-title">🏛 DMU Analytics</div>
          <div className="sidebar-sub">Nilgiris Educational Intelligence</div>
        </div>

        <div className="sidebar-section">Examples</div>
        {EXAMPLES.map((q, i) => (
          <button key={i} className="example-btn" onClick={() => setInput(q)}>
            {q}
          </button>
        ))}

        <div className="sidebar-section" style={{ marginTop: 8 }}>History</div>
        <div className="history-list">
          {history.length === 0 && (
            <div style={{ fontSize: 11, color: '#6e7681', padding: '8px 10px' }}>
              No queries yet
            </div>
          )}
          {[...history].reverse().map((m, i) => (
            <div key={i} className="history-item" onClick={() => setInput(m.content)}>
              <div className="history-q">{m.content}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Main area */}
      <div className="main">
        {/* Topbar */}
        <div className="topbar">
          <div className="topbar-title">Educational Decision Intelligence</div>
          <div className="topbar-metrics">
            {health && (
              <span className={`metric-chip ${health.status === 'ok' ? 'green' : ''}`}>
                DB {health.status === 'ok' ? '●' : '○'} {health.marks_rows?.toLocaleString()} rows
              </span>
            )}
            {metrics && (
              <>
                <span className="metric-chip blue">
                  {metrics.total_requests} queries
                </span>
                <span className="metric-chip">
                  ${(metrics.total_cost_usd || 0).toFixed(4)} cost
                </span>
                <span
                  className="metric-chip"
                  style={{ cursor: 'pointer' }}
                  onClick={() => setShowMetrics(v => !v)}
                >
                  📊 Metrics
                </span>
              </>
            )}
          </div>
        </div>

        {/* Chat area */}
        <div className="chat-area">
          {messages.length === 0 ? (
            <div className="empty-state">
              <div className="empty-icon">📊</div>
              <div className="empty-title">Ask anything about Nilgiris student performance</div>
              <div className="empty-sub">
                Multi-agent AI: SQL analytics, document RAG, web search, visualizations,
                and actionable recommendations — all in one place.
              </div>
            </div>
          ) : (
            <>
              {messages.map((m, i) => <Message key={i} msg={m} />)}
              {loading && (
                <div className="msg">
                  <div className="msg-assistant">
                    <div className="msg-avatar">D</div>
                    <div className="msg-content">
                      <LoadingDots />
                    </div>
                  </div>
                </div>
              )}
              {showMetrics && metrics && (
                <MetricsPanel metrics={metrics} />
              )}
            </>
          )}
          <div ref={bottomRef} />
        </div>

        {/* Input bar */}
        <div className="input-bar">
          <div className="input-row">
            <div className="input-wrap">
              <textarea
                ref={textareaRef}
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={onKey}
                placeholder="Ask about student performance, school rankings, subject failures… (Enter to send)"
                rows={1}
              />
            </div>
            <button className="send-btn" onClick={send} disabled={loading || !input.trim()}>
              {loading ? <div className="spinner" /> : '→ Send'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
