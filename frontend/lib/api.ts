const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080";

export interface FlaggedAccount {
  account_id: string;
  risk_score: number;
  detected_patterns: string[];
  pattern_agent_summary: string;
  risk_agent_summary: string;
  investigator_explanation: string;
  graph_connections: { from: string; to: string; amount: number }[];
}

export interface PipelineResult {
  flagged_accounts: FlaggedAccount[];
  graph: GraphData;
  meta: { total_flagged: number; total_nodes: number; total_edges: number };
}

export interface AccountResult {
  account_id: number;
  flag: "NORMAL" | "SUSPICIOUS" | "LAUNDERING";
  aiExplanation: string;
}

export interface GraphNode {
  id: string;
  label: string;
  type?: string;
}

export interface GraphEdge {
  from: string;
  to: string;
  amount?: number;
  source?: string;
  target?: string;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface HealthStatus {
  status: string;
  db: string;
  model_configured: boolean;
}

async function apiFetch<T>(path: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, opts);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status}: ${body}`);
  }
  return res.json();
}

export async function runPipeline(): Promise<PipelineResult> {
  return apiFetch<PipelineResult>("/pipeline/run", { method: "POST" });
}

export async function getFlagged(): Promise<FlaggedAccount[]> {
  return apiFetch<FlaggedAccount[]>("/flagged");
}

export async function getAccount(id: string): Promise<AccountResult> {
  return apiFetch<AccountResult>(`/accounts/${id}`);
}

export async function getGraph(id?: string): Promise<GraphData> {
  const path = id ? `/graph/${id}` : "/graph/";
  return apiFetch<GraphData>(path);
}

export async function getHealth(): Promise<HealthStatus> {
  return apiFetch<HealthStatus>("/health");
}
