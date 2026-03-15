export interface GraphNode {
  id: string;
  risk: 'normal' | 'suspicious' | 'laundering';
  txCount: number;
  pattern: string;
  aiExplanation: string;
  role?: string;
  riskScore?: number;
  cluster?: number;
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

function parseCSV(text: string): Record<string, string>[] {
  const parsedRows = parseCsvRows(text);
  if (parsedRows.length < 2) return [];

  const [headers, ...dataRows] = parsedRows;
  const result: Record<string, string>[] = [];

  for (const values of dataRows) {
    const row: Record<string, string> = {};
    for (let j = 0; j < headers.length; j++) {
      row[headers[j]] = values[j] ?? '';
    }
    result.push(row);
  }
  return result;
}

function parseCsvRows(text: string): string[][] {
  const rows: string[][] = [];
  const input = text.replace(/^\uFEFF/, '');
  let currentRow: string[] = [];
  let current = '';
  let inQuotes = false;

  for (let i = 0; i < input.length; i++) {
    const ch = input[i];

    if (inQuotes) {
      if (ch === '"') {
        if (i + 1 < input.length && input[i + 1] === '"') {
          current += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        current += ch;
      }
    } else {
      if (ch === '"') {
        inQuotes = true;
      } else if (ch === ',') {
        currentRow.push(current);
        current = '';
      } else if (ch === '\n') {
        currentRow.push(current);
        rows.push(currentRow);
        currentRow = [];
        current = '';
      } else if (ch === '\r') {
        if (input[i + 1] === '\n') continue;
        currentRow.push(current);
        rows.push(currentRow);
        currentRow = [];
        current = '';
      } else {
        current += ch;
      }
    }
  }

  if (current.length > 0 || currentRow.length > 0) {
    currentRow.push(current);
    rows.push(currentRow);
  }

  return rows.filter((row) => row.some((value) => value.length > 0));
}

function toRisk(value: string): 'normal' | 'suspicious' | 'laundering' {
  if (value === 'laundering' || value === 'suspicious') return value;
  return 'normal';
}

const CSV_BASE_PATHS = ['/data', '/node_data'] as const;

async function loadCsvText(fileName: 'nodes.csv' | 'edges.csv'): Promise<string> {
  for (const basePath of CSV_BASE_PATHS) {
    const response = await fetch(`${basePath}/${fileName}`);
    if (response.ok) {
      return response.text();
    }
  }

  throw new Error(`Could not load ${fileName}`);
}

export async function loadGraphFromCSV(): Promise<GraphData> {
  const [nodesText, edgesText] = await Promise.all([
    loadCsvText('nodes.csv'),
    loadCsvText('edges.csv'),
  ]);

  const nodesRaw = parseCSV(nodesText);
  const edgesRaw = parseCSV(edgesText);

  const nodes: GraphNode[] = nodesRaw.map((r) => ({
    id: r.id,
    risk: toRisk(r.risk),
    txCount: parseInt(r.txCount, 10) || 0,
    pattern: r.pattern || 'None',
    aiExplanation: r.aiExplanation || 'No anomalies detected.',
    role: r.role || undefined,
    riskScore: r.riskScore ? parseFloat(r.riskScore) : undefined,
    cluster: r.cluster ? parseInt(r.cluster, 10) : undefined,
  }));

  const links: GraphLink[] = edgesRaw.map((r) => ({
    source: r.source,
    target: r.target,
    amount: parseFloat(r.amount) || 0,
  }));

  return { nodes, links };
}
