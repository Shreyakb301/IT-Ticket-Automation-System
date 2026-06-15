import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

function AnalyticsCharts({ analytics }) {
  const categoryData = analytics ? Object.entries(analytics.category_counts).map(([name, value]) => ({ name, value })) : [];
  const priorityData = analytics ? Object.entries(analytics.priority_counts).map(([name, value]) => ({ name, value })) : [];

  return (
    <section className="grid two" aria-label="Ticket analytics charts">
      <div className="card chartCard">
        <p className="eyebrow">ANALYTICS</p>
        <h2>Tickets by model category</h2>
        <div role="img" aria-label="Bar chart showing ticket counts by model category">
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={categoryData} margin={{ top: 8, right: 12, bottom: 72, left: 0 }}>
              <XAxis dataKey="name" interval={0} angle={-35} textAnchor="end" height={78} tick={{ fontSize: 12 }} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="value" fill="#1d4ed8" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
      <div className="card chartCard">
        <p className="eyebrow">ANALYTICS</p>
        <h2>Priority distribution</h2>
        <div role="img" aria-label="Pie chart showing ticket priority distribution">
          <ResponsiveContainer width="100%" height={260}>
            <PieChart>
              <Pie data={priorityData} dataKey="value" nameKey="name" outerRadius={90} label>
                {priorityData.map((entry, index) => <Cell key={entry.name} fill={COLORS[index % COLORS.length]} />)}
              </Pie>
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>
    </section>
  );
}

const COLORS = ['#1d4ed8', '#047857', '#b45309', '#7c3aed', '#be123c'];

export default AnalyticsCharts;
