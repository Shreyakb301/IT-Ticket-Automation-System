import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { Brain, Ticket, Activity, CheckCircle2 } from 'lucide-react';
import './style.css';

const API = 'http://localhost:8000';

function asPercent(value) {
  return `${Math.round((value || 0) * 100)}%`;
}

function App() {
  const [ticketText, setTicketText] = useState('My VPN disconnects every few minutes when I work from home.');
  const [prediction, setPrediction] = useState(null);
  const [analytics, setAnalytics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    fetch(`${API}/analytics`)
      .then((res) => res.json())
      .then(setAnalytics)
      .catch(() => setAnalytics(null));
  }, []);

  async function classifyTicket() {
    setLoading(true);
    setError('');
    try {
      const res = await fetch(`${API}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticket_text: ticketText }),
      });
      if (!res.ok) throw new Error('Prediction failed. Make sure the FastAPI backend is running and models are trained.');
      setPrediction(await res.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const categoryData = analytics ? Object.entries(analytics.category_counts).map(([name, value]) => ({ name, value })) : [];
  const priorityData = analytics ? Object.entries(analytics.priority_counts).map(([name, value]) => ({ name, value })) : [];

  return (
    <main className="page">
      <section className="hero card">
        <div>
          <p className="eyebrow">NLP CLASSIFICATION SYSTEM</p>
          <h1>IT Ticket Automated Classifier</h1>
          <p className="subtitle">Classifies support tickets by category, subcategory, and priority using Sentence Transformers + XGBoost.</p>
        </div>
        <div className="heroIcon"><Brain size={42} /></div>
      </section>

      <section className="grid two">
        <div className="card">
          <p className="eyebrow">REAL-TIME TRIAGE</p>
          <h2>Submit a ticket</h2>
          <textarea value={ticketText} onChange={(e) => setTicketText(e.target.value)} />
          <button onClick={classifyTicket} disabled={loading}>{loading ? 'Classifying...' : 'Classify Ticket'}</button>
          {error && <p className="error">{error}</p>}
        </div>

        <div className="card result">
          <p className="eyebrow">MODEL OUTPUT</p>
          <h2>Prediction</h2>
          {!prediction ? <p className="muted">Run a prediction to see classification results.</p> : (
            <div className="predictionGrid">
              <Result label="Category" value={prediction.category} confidence={prediction.category_confidence} />
              <Result label="Subcategory" value={prediction.subcategory} confidence={prediction.subcategory_confidence} />
              <Result label="Priority" value={prediction.priority} confidence={prediction.priority_confidence} />
            </div>
          )}
        </div>
      </section>

      <section className="stats grid four">
        <Metric icon={<Ticket />} label="Dataset" value={analytics ? analytics.total_tickets.toLocaleString() : '10,000'} />
        <Metric icon={<Activity />} label="Targets" value="3" />
        <Metric icon={<CheckCircle2 />} label="Embedding Size" value="384" />
        <Metric icon={<Brain />} label="Model" value="MiniLM + XGB" />
      </section>

      <section className="grid two">
        <div className="card chartCard">
          <p className="eyebrow">ANALYTICS</p>
          <h2>Tickets by category</h2>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={categoryData}><XAxis dataKey="name" /><YAxis /><Tooltip /><Bar dataKey="value" /></BarChart>
          </ResponsiveContainer>
        </div>
        <div className="card chartCard">
          <p className="eyebrow">ANALYTICS</p>
          <h2>Priority distribution</h2>
          <ResponsiveContainer width="100%" height={260}>
            <PieChart><Pie data={priorityData} dataKey="value" nameKey="name" outerRadius={90} label>{priorityData.map((_, i) => <Cell key={i} />)}</Pie></PieChart>
          </ResponsiveContainer>
        </div>
      </section>
    </main>
  );
}

function Result({ label, value, confidence }) {
  return <div className="resultBox"><span>{label}</span><strong>{value}</strong><small>Confidence {asPercent(confidence)}</small></div>;
}

function Metric({ icon, label, value }) {
  return <div className="card metric"><div>{icon}</div><span>{label}</span><strong>{value}</strong></div>;
}

createRoot(document.getElementById('root')).render(<App />);
