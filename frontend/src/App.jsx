import React, { Suspense, lazy, useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { AlertTriangle, Brain, CheckCircle2, PlayCircle, Route, ShieldCheck, Ticket } from 'lucide-react';
import './style.css';

const AnalyticsCharts = lazy(() => import('./AnalyticsCharts.jsx'));
const API = import.meta.env.VITE_API_URL || (import.meta.env.DEV ? 'http://localhost:8000' : '');
const SAMPLE_PROMPTS = [
  'My VPN disconnects every few minutes when I work from home.',
  'Cannot log in after password reset and MFA keeps failing.',
  'Outlook keeps crashing whenever I open email attachments.',
  'Printer says the job completed but nothing printed.',
  'Postgres database connection keeps timing out during checkout reports.',
  'New laptop will not turn on after docking station update.',
  'Suspicious login alert from another country on my account.',
  'Company WiFi is connected but internal apps will not load.',
];
const SUPPORT_GROUP_COUNT = 8;

function asPercent(value) {
  return `${Math.round((value || 0) * 100)}%`;
}

function App() {
  const [ticketText, setTicketText] = useState('My VPN disconnects every few minutes when I work from home.');
  const [prediction, setPrediction] = useState(null);
  const [analytics, setAnalytics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [promptIndex, setPromptIndex] = useState(0);
  const [demoVisible, setDemoVisible] = useState(false);

  useEffect(() => {
    if (!demoVisible || analytics) return;
    fetch(`${API}/analytics`)
      .then((res) => res.json())
      .then(setAnalytics)
      .catch(() => setAnalytics(null));
  }, [analytics, demoVisible]);

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

  function changePrompt() {
    const nextIndex = (promptIndex + 1) % SAMPLE_PROMPTS.length;
    setPromptIndex(nextIndex);
    setTicketText(SAMPLE_PROMPTS[nextIndex]);
    setPrediction(null);
    setError('');
  }

  function showDemo() {
    setDemoVisible(true);
    window.setTimeout(() => {
      document.getElementById('demo')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 80);
  }

  return (
    <main id="main-content" className="page">
      <section className="landing">
        <div>
          <p className="eyebrow">HELPDESK ROUTING SYSTEM</p>
          <h1>IT Ticket Routing Automation</h1>
          <p className="subtitle">Reads incoming Helpdesk tickets and recommends the right IT support group, issue type, and priority with benchmarked NLP models, confidence scores, and human-review routing.</p>
          <div className="landingActions">
            <button onClick={showDemo} aria-label="View ticket routing demo"><PlayCircle size={18} /> View Demo</button>
          </div>
        </div>
        <div className="workflowPanel" aria-label="Routing workflow summary">
          <div><Ticket /><span>Incoming Ticket</span></div>
          <div><Brain /><span>NLP Prediction</span></div>
          <div><Route /><span>Support Group</span></div>
          <div><ShieldCheck /><span>Confidence Check</span></div>
        </div>
      </section>

      <section id="demo" className={`demoSection ${demoVisible ? 'show' : ''}`}>
      <section className="grid two">
        <div className="card triageCard">
          <p className="eyebrow">REAL-TIME TRIAGE</p>
          <h2>Submit a Helpdesk ticket</h2>
          <label className="fieldLabel" htmlFor="ticketText">Ticket text</label>
          <textarea id="ticketText" value={ticketText} onChange={(e) => setTicketText(e.target.value)} />
          <div className="buttonRow">
            <button onClick={classifyTicket} disabled={loading} aria-label="Route ticket">{loading ? 'Routing...' : 'Route Ticket'}</button>
            <button className="secondaryButton" onClick={changePrompt} disabled={loading} aria-label="Change sample ticket prompt">Change Prompt</button>
          </div>
          {error && <p className="error">{error}</p>}
        </div>

        <div className="card result">
          <p className="eyebrow">MODEL OUTPUT</p>
          <h2>Prediction</h2>
          {!prediction ? <p className="muted">Run a ticket to see the recommended support group and confidence.</p> : (
            <>
              <RoutingDecision prediction={prediction} />
              <div className="predictionGrid">
                <Result label="Support Group" value={prediction.support_group || prediction.category} confidence={prediction.support_group_confidence || prediction.category_confidence} />
                <Result label="Issue Type" value={prediction.issue_type || prediction.subcategory} confidence={prediction.issue_type_confidence || prediction.subcategory_confidence} />
                <Result label="Priority" value={prediction.priority} confidence={prediction.priority_confidence} />
              </div>
            </>
          )}
        </div>
      </section>

      <section className="stats grid four">
        <Metric icon={<Ticket />} label="Dataset" value={analytics ? analytics.total_tickets.toLocaleString() : '20,000'} />
        <Metric icon={<Route />} label="IT Groups" value={SUPPORT_GROUP_COUNT} />
        <Metric icon={<CheckCircle2 />} label="Best Accuracy" value="81.35%" />
        <Metric icon={<Brain />} label="Best Model" value="DistilBERT" />
      </section>

      <Suspense fallback={<p className="muted">Loading analytics...</p>}>
        <AnalyticsCharts analytics={analytics} />
      </Suspense>
      </section>
    </main>
  );
}

function Result({ label, value, confidence }) {
  return <div className="resultBox"><span>{label}</span><strong>{value}</strong><small>Confidence {asPercent(confidence)}</small></div>;
}

function RoutingDecision({ prediction }) {
  const isAuto = Boolean(prediction.auto_route);
  return (
    <div className={`decision ${isAuto ? 'auto' : 'review'}`}>
      <div className="decisionIcon">{isAuto ? <ShieldCheck size={22} /> : <AlertTriangle size={22} />}</div>
      <div>
        <span>{isAuto ? 'AUTO-ROUTE' : 'TRIAGE REVIEW'}</span>
        <strong>{prediction.routing_decision || (isAuto ? 'Auto-route' : 'Human review required')}</strong>
        {prediction.review_reason && <small>{prediction.review_reason}</small>}
      </div>
      <Route size={18} />
    </div>
  );
}

function Metric({ icon, label, value }) {
  return <div className="card metric"><div>{icon}</div><span>{label}</span><strong>{value}</strong></div>;
}

createRoot(document.getElementById('root')).render(<App />);
