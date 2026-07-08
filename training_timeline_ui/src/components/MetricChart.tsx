import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { MetricSeries } from "../types";

type MetricChartProps = {
  title: string;
  series: MetricSeries;
  events?: Array<{ start_step?: number | null; end_step?: number | null; title: string }>;
};

export function MetricChart({ title, series }: MetricChartProps) {
  const names = Object.keys(series).filter((name) => series[name]?.length);
  const data = mergeSeries(series, names);

  return (
    <div className="chart-panel">
      <h3>{title}</h3>
      {data.length === 0 ? (
        <div className="empty-chart">No curve data</div>
      ) : (
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={data} margin={{ top: 12, right: 18, left: 4, bottom: 6 }}>
            <CartesianGrid stroke="#e6ecf2" />
            <XAxis dataKey="step" tick={{ fontSize: 12 }} />
            <YAxis tick={{ fontSize: 12 }} width={72} tickFormatter={formatAxisTick} />
            <Tooltip />
            {names.map((name, index) => (
              <Line key={name} type="monotone" dataKey={name} stroke={COLORS[index % COLORS.length]} dot={false} strokeWidth={2} connectNulls />
            ))}
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}

const COLORS = ["#2f6f8f", "#8a5a44", "#4c7f47", "#7a4e8a", "#a06c2f", "#516a9b"];

function formatAxisTick(value: number) {
  const magnitude = Math.abs(value);
  if (magnitude > 0 && magnitude < 0.001) {
    return value.toExponential(1);
  }
  return Number.isInteger(value) ? String(value) : value.toPrecision(3);
}

function mergeSeries(series: MetricSeries, names: string[]) {
  const byStep = new Map<number, Record<string, number>>();
  for (const name of names) {
    for (const point of series[name] ?? []) {
      const row = byStep.get(point.step) ?? { step: point.step };
      row[name] = point.value;
      byStep.set(point.step, row);
    }
  }
  return [...byStep.values()].sort((a, b) => Number(a.step) - Number(b.step));
}
