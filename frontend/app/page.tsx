"use client";

import { useState, useEffect, useCallback } from "react";
import {
  runPipeline,
  getFlagged,
  getAccount,
  getGraph,
  getHealth,
  type FlaggedAccount,
  type AccountResult,
  type GraphData,
  type HealthStatus,
} from "@/lib/api";

/* ------------------------------------------------------------------ */
/*  Risk badge                                                        */
/* ------------------------------------------------------------------ */
function RiskBadge({ score }: { score: number }) {
  let color = "bg-green-600";
  let label = "Low";
  if (score >= 0.9) {
    color = "bg-red-600";
    label = "Critical";
  } else if (score >= 0.7) {
    color = "bg-orange-500";
    label = "High";
  } else if (score >= 0.5) {
    color = "bg-yellow-500 text-black";
    label = "Medium";
  }
  return (
    <span className={`${color} px-2 py-0.5 rounded text-xs font-semibold`}>
      {label} ({score.toFixed(2)})
    </span>
  );
}

function FlagBadge({ flag }: { flag: string }) {
  const colors: Record<string, string> = {
    LAUNDERING: "bg-red-600",
    SUSPICIOUS: "bg-orange-500",
    NORMAL: "bg-green-600",
  };
  return (
    <span
      className={`${colors[flag] || "bg-gray-500"} px-2 py-0.5 rounded text-xs font-bold`}
    >
      {flag}
    </span>
  );
}

/* ------------------------------------------------------------------ */
/*  Pattern pills                                                     */
/* ------------------------------------------------------------------ */
function PatternPills({ patterns }: { patterns: string[] }) {
  if (!patterns.length) return <span className="text-slate-500">none</span>;
  return (
    <div className="flex gap-1 flex-wrap">
      {patterns.map((p) => (
        <span
          key={p}
          className="bg-slate-700 text-slate-200 px-2 py-0.5 rounded text-xs"
        >
          {p}
        </span>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Simple graph canvas (SVG force-ish layout)                        */
/* ------------------------------------------------------------------ */
function GraphView({ data }: { data: GraphData }) {
  if (!data.nodes.length)
    return <p className="text-slate-500">No graph data.</p>;

  const W = 700;
  const H = 400;
  const nodeMap = new Map<string, { x: number; y: number }>();

  // Assign positions in a circle
  data.nodes.forEach((n, i) => {
    const angle = (2 * Math.PI * i) / data.nodes.length;
    const r = Math.min(W, H) * 0.38;
    nodeMap.set(n.id, {
      x: W / 2 + r * Math.cos(angle),
      y: H / 2 + r * Math.sin(angle),
    });
  });

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="w-full max-h-[400px] bg-slate-900 rounded-lg border border-slate-700"
    >
      {data.edges.map((e, i) => {
        const src = nodeMap.get(e.from || e.source || "");
        const dst = nodeMap.get(e.to || e.target || "");
        if (!src || !dst) return null;
        return (
          <line
            key={i}
            x1={src.x}
            y1={src.y}
            x2={dst.x}
            y2={dst.y}
            stroke="#475569"
            strokeWidth={0.5}
            opacity={0.6}
          />
        );
      })}
      {data.nodes.map((n) => {
        const pos = nodeMap.get(n.id)!;
        return (
          <g key={n.id}>
            <circle cx={pos.x} cy={pos.y} r={4} fill="#3b82f6" />
            <text
              x={pos.x + 6}
              y={pos.y + 3}
              fontSize={8}
              fill="#94a3b8"
            >
              {n.label.length > 10 ? n.label.slice(0, 10) + ".." : n.label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/* ------------------------------------------------------------------ */
/*  Account detail panel                                              */
/* ------------------------------------------------------------------ */
function AccountPanel({
  accountId,
  onClose,
}: {
  accountId: string;
  onClose: () => void;
}) {
  const [data, setData] = useState<AccountResult | null>(null);
  const [graph, setGraph] = useState<GraphData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    setLoading(true);
    setError("");
    Promise.all([getAccount(accountId), getGraph(accountId)])
      .then(([acc, g]) => {
        setData(acc);
        setGraph(g);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [accountId]);

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4">
      <div className="bg-card border border-card-border rounded-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto p-6">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-bold">Account {accountId}</h2>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white text-2xl leading-none"
          >
            &times;
          </button>
        </div>

        {loading && <p className="text-slate-400">Loading...</p>}
        {error && <p className="text-red-400">{error}</p>}

        {data && (
          <div className="space-y-4">
            <div className="flex gap-3 items-center">
              <FlagBadge flag={data.flag} />
              <span className="text-sm text-slate-400">
                Account #{data.account_id}
              </span>
            </div>

            <div>
              <h3 className="text-sm font-semibold text-slate-300 mb-1">
                Watsonx AI Explanation
              </h3>
              <p className="text-sm text-slate-300 bg-slate-800 rounded p-3 leading-relaxed">
                {data.aiExplanation}
              </p>
            </div>

            {graph && graph.nodes.length > 0 && (
              <div>
                <h3 className="text-sm font-semibold text-slate-300 mb-1">
                  Account Subgraph ({graph.nodes.length} nodes,{" "}
                  {graph.edges.length} edges)
                </h3>
                <GraphView data={graph} />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Main dashboard                                                    */
/* ------------------------------------------------------------------ */
export default function Dashboard() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [flagged, setFlagged] = useState<FlaggedAccount[]>([]);
  const [graph, setGraph] = useState<GraphData | null>(null);
  const [selectedAccount, setSelectedAccount] = useState<string | null>(null);

  const [pipelineRunning, setPipelineRunning] = useState(false);
  const [pipelineError, setPipelineError] = useState("");
  const [pipelineDone, setPipelineDone] = useState(false);
  const [loading, setLoading] = useState(true);

  // Check health + try to load cached results on mount
  useEffect(() => {
    Promise.all([
      getHealth().catch(() => null),
      getFlagged().catch(() => []),
    ]).then(([h, f]) => {
      if (h) setHealth(h);
      if (Array.isArray(f) && f.length) {
        setFlagged(f);
        setPipelineDone(true);
        getGraph().then(setGraph).catch(() => {});
      }
      setLoading(false);
    });
  }, []);

  const handleRunPipeline = useCallback(async () => {
    setPipelineRunning(true);
    setPipelineError("");
    setPipelineDone(false);
    try {
      const result = await runPipeline();
      setFlagged(result.flagged_accounts);
      setGraph(result.graph);
      setPipelineDone(true);
      // refresh health
      getHealth().then(setHealth).catch(() => {});
    } catch (e: unknown) {
      setPipelineError(e instanceof Error ? e.message : String(e));
    } finally {
      setPipelineRunning(false);
    }
  }, []);

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="border-b border-card-border bg-card">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold tracking-tight">
              GenAI Genesis
            </h1>
            <p className="text-xs text-slate-400">
              GNN + Railtracks + Watsonx AML Detection
            </p>
          </div>
          <div className="flex items-center gap-4">
            {health && (
              <span
                className={`text-xs px-2 py-1 rounded ${
                  health.status === "ok"
                    ? "bg-green-900 text-green-300"
                    : "bg-yellow-900 text-yellow-300"
                }`}
              >
                API: {health.status} | DB: {health.db} | Model:{" "}
                {health.model_configured ? "ready" : "missing"}
              </span>
            )}
            <button
              onClick={handleRunPipeline}
              disabled={pipelineRunning}
              className="bg-accent hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold px-4 py-2 rounded-lg text-sm transition-colors"
            >
              {pipelineRunning ? "Running Pipeline..." : "Run Pipeline"}
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-6 space-y-6">
        {/* Pipeline status */}
        {pipelineError && (
          <div className="bg-red-900/40 border border-red-700 rounded-lg p-4 text-sm text-red-300">
            Pipeline error: {pipelineError}
          </div>
        )}

        {loading && (
          <p className="text-slate-400 text-center py-12">
            Connecting to backend...
          </p>
        )}

        {!loading && !pipelineDone && !pipelineRunning && (
          <div className="text-center py-16 space-y-3">
            <p className="text-slate-400 text-lg">No pipeline results yet.</p>
            <p className="text-slate-500 text-sm">
              Click <strong>Run Pipeline</strong> to load data, run GNN
              inference, and generate AI explanations.
            </p>
          </div>
        )}

        {pipelineRunning && (
          <div className="text-center py-16 space-y-3">
            <div className="inline-block w-8 h-8 border-4 border-accent border-t-transparent rounded-full animate-spin" />
            <p className="text-slate-400">
              Running: Load &rarr; Preprocess &rarr; GNN &rarr; Railtracks
              Agents &rarr; Watsonx...
            </p>
          </div>
        )}

        {/* Results */}
        {pipelineDone && flagged.length > 0 && (
          <>
            {/* Stats row */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div className="bg-card border border-card-border rounded-lg p-4">
                <p className="text-xs text-slate-400 uppercase tracking-wide">
                  Flagged Accounts
                </p>
                <p className="text-2xl font-bold mt-1">{flagged.length}</p>
              </div>
              <div className="bg-card border border-card-border rounded-lg p-4">
                <p className="text-xs text-slate-400 uppercase tracking-wide">
                  Critical (score &ge; 0.9)
                </p>
                <p className="text-2xl font-bold text-red-400 mt-1">
                  {flagged.filter((a) => a.risk_score >= 0.9).length}
                </p>
              </div>
              <div className="bg-card border border-card-border rounded-lg p-4">
                <p className="text-xs text-slate-400 uppercase tracking-wide">
                  Graph Nodes / Edges
                </p>
                <p className="text-2xl font-bold mt-1">
                  {graph?.nodes.length ?? 0} / {graph?.edges.length ?? 0}
                </p>
              </div>
            </div>

            {/* Investigator summary (shared across flagged) */}
            {flagged[0]?.investigator_explanation && (
              <div className="bg-card border border-card-border rounded-lg p-4">
                <h2 className="text-sm font-semibold text-slate-300 mb-2">
                  Investigator Summary (Railtracks)
                </h2>
                <p className="text-sm text-slate-300 leading-relaxed">
                  {flagged[0].investigator_explanation}
                </p>
              </div>
            )}

            {/* Agent summaries */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {flagged[0]?.pattern_agent_summary && (
                <div className="bg-card border border-card-border rounded-lg p-4">
                  <h3 className="text-xs font-semibold text-blue-400 uppercase tracking-wide mb-2">
                    Pattern Agent
                  </h3>
                  <p className="text-sm text-slate-300 leading-relaxed">
                    {flagged[0].pattern_agent_summary}
                  </p>
                </div>
              )}
              {flagged[0]?.risk_agent_summary && (
                <div className="bg-card border border-card-border rounded-lg p-4">
                  <h3 className="text-xs font-semibold text-orange-400 uppercase tracking-wide mb-2">
                    Risk Agent
                  </h3>
                  <p className="text-sm text-slate-300 leading-relaxed">
                    {flagged[0].risk_agent_summary}
                  </p>
                </div>
              )}
            </div>

            {/* Flagged accounts table */}
            <div className="bg-card border border-card-border rounded-lg overflow-hidden">
              <div className="px-4 py-3 border-b border-card-border">
                <h2 className="font-semibold">Flagged Accounts</h2>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-card-border text-slate-400 text-xs uppercase tracking-wide">
                      <th className="text-left px-4 py-2">Account</th>
                      <th className="text-left px-4 py-2">Risk</th>
                      <th className="text-left px-4 py-2">Patterns</th>
                      <th className="text-left px-4 py-2">Connections</th>
                      <th className="text-left px-4 py-2">Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {flagged.map((acc) => (
                      <tr
                        key={acc.account_id}
                        className="border-b border-card-border/50 hover:bg-slate-800/50 transition-colors"
                      >
                        <td className="px-4 py-3 font-mono">
                          {acc.account_id}
                        </td>
                        <td className="px-4 py-3">
                          <RiskBadge score={acc.risk_score} />
                        </td>
                        <td className="px-4 py-3">
                          <PatternPills patterns={acc.detected_patterns} />
                        </td>
                        <td className="px-4 py-3 text-slate-400">
                          {acc.graph_connections.length} edges
                        </td>
                        <td className="px-4 py-3">
                          <button
                            onClick={() =>
                              setSelectedAccount(acc.account_id)
                            }
                            className="text-accent hover:text-blue-300 text-xs font-semibold"
                          >
                            View Details
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Full graph */}
            {graph && graph.nodes.length > 0 && (
              <div className="bg-card border border-card-border rounded-lg p-4">
                <h2 className="font-semibold mb-3">
                  Transaction Graph ({graph.nodes.length} nodes,{" "}
                  {graph.edges.length} edges)
                </h2>
                <GraphView data={graph} />
              </div>
            )}
          </>
        )}

        {pipelineDone && flagged.length === 0 && (
          <div className="text-center py-12">
            <p className="text-slate-400">
              Pipeline completed. No accounts flagged above the risk threshold.
            </p>
          </div>
        )}
      </main>

      {/* Account detail modal */}
      {selectedAccount && (
        <AccountPanel
          accountId={selectedAccount}
          onClose={() => setSelectedAccount(null)}
        />
      )}
    </div>
  );
}
