const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';

export interface GraphNode {
  id: string;
  risk: 'normal' | 'suspicious' | 'laundering';
  txCount: number;
  pattern: string;
  aiExplanation: string;
  role?: string;
  riskScore?: number;
}

export interface GraphLink {
  source: string;
  target: string;
  amount: number;
}

export interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}

interface FlaggedAccount {
  account_id: string;
  risk_score: number;
  detected_patterns: string[];
  investigator_explanation: string;
  role: string;
  fan_in: number;
  fan_out: number;
}

interface BackendGraph {
  nodes: { id: string; label?: string; [key: string]: any }[];
  edges: { from: string; to: string; amount: number; [key: string]: any }[];
}

function riskLevel(score: number): 'normal' | 'suspicious' | 'laundering' {
  if (score >= 0.9) return 'laundering';
  if (score >= 0.7) return 'suspicious';
  return 'normal';
}

export async function runPipeline(): Promise<GraphData> {
  const res = await fetch(`${API_BASE}/pipeline/run`, { method: 'POST' });
  if (!res.ok) throw new Error(`Pipeline failed: ${res.status}`);
  const data = await res.json();

  const flaggedMap = new Map<string, FlaggedAccount>();
  for (const f of data.flagged_accounts ?? []) {
    flaggedMap.set(String(f.account_id), f);
  }

  const riskScores: Record<string, number> = data.account_risk_scores ?? {};
  const graph: BackendGraph = data.graph ?? { nodes: [], edges: [] };

  const nodes: GraphNode[] = graph.nodes.map((n) => {
    const id = String(n.id);
    const flagged = flaggedMap.get(id);
    const score = riskScores[id] ?? riskScores[n.id] ?? 0;
    return {
      id,
      risk: riskLevel(score),
      txCount: (flagged?.fan_in ?? 0) + (flagged?.fan_out ?? 0),
      pattern: flagged?.detected_patterns?.join(', ') || 'None',
      aiExplanation: flagged?.investigator_explanation || 'No anomalies detected.',
      role: flagged?.role,
      riskScore: score,
    };
  });

  const links: GraphLink[] = graph.edges.map((e) => ({
    source: String(e.from),
    target: String(e.to),
    amount: e.amount,
  }));

  return { nodes, links };
}

export async function fetchGraph(): Promise<GraphData> {
  const [graphRes, flaggedRes] = await Promise.all([
    fetch(`${API_BASE}/graph`),
    fetch(`${API_BASE}/flagged`),
  ]);

  if (!graphRes.ok) throw new Error(`Graph fetch failed: ${graphRes.status}`);

  const graph: BackendGraph = await graphRes.json();
  const flagged: FlaggedAccount[] = flaggedRes.ok ? await flaggedRes.json() : [];

  const flaggedMap = new Map<string, FlaggedAccount>();
  for (const f of flagged) {
    flaggedMap.set(String(f.account_id), f);
  }

  const nodes: GraphNode[] = graph.nodes.map((n) => {
    const id = String(n.id);
    const f = flaggedMap.get(id);
    const score = f?.risk_score ?? 0;
    return {
      id,
      risk: riskLevel(score),
      txCount: (f?.fan_in ?? 0) + (f?.fan_out ?? 0),
      pattern: f?.detected_patterns?.join(', ') || 'None',
      aiExplanation: f?.investigator_explanation || 'No anomalies detected.',
      role: f?.role,
      riskScore: score,
    };
  });

  const links: GraphLink[] = graph.edges.map((e) => ({
    source: String(e.from),
    target: String(e.to),
    amount: e.amount,
  }));

  return { nodes, links };
}
