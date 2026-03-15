'use client';

import { useRef, useEffect, useState, useMemo, useCallback } from 'react';
import ForceGraph2D from 'react-force-graph-2d';

// --- Cluster color palette ---
const CLUSTER_COLORS = [
  '#ff6b35', '#1dd3b0', '#3772ff', '#ef476f',
  '#ffd166', '#06d6a0', '#118ab2', '#9b5de5',
  '#f15bb5', '#00bbf9', '#fee440', '#8ac926',
];

const RISK_NODE_COLORS = {
  normal: '#3772ff',
  suspicious: '#f97316',
  laundering: '#ef4444',
} as const;

// --- BFS connected-component clustering ---
function computeClusters(nodes: any[], links: any[]) {
  const adj = new Map<string, Set<string>>();
  for (const n of nodes) adj.set(n.id, new Set());
  for (const l of links) {
    const s = typeof l.source === 'object' ? l.source.id : l.source;
    const t = typeof l.target === 'object' ? l.target.id : l.target;
    adj.get(s)?.add(t);
    adj.get(t)?.add(s);
  }

  const clusterMap = new Map<string, number>();
  let clusterId = 0;
  const visited = new Set<string>();

  // Sort by degree descending so the biggest component gets cluster 0
  const sortedNodes = [...nodes].sort((a, b) => (adj.get(b.id)?.size ?? 0) - (adj.get(a.id)?.size ?? 0));

  for (const node of sortedNodes) {
    if (visited.has(node.id)) continue;
    const queue = [node.id];
    visited.add(node.id);
    while (queue.length > 0) {
      const current = queue.shift()!;
      clusterMap.set(current, clusterId);
      for (const neighbor of adj.get(current) ?? []) {
        if (!visited.has(neighbor)) {
          visited.add(neighbor);
          queue.push(neighbor);
        }
      }
    }
    clusterId++;
  }

  return clusterMap;
}

// --- Compute node degrees ---
function computeDegrees(nodes: any[], links: any[]) {
  const degreeMap = new Map<string, number>();
  for (const n of nodes) degreeMap.set(n.id, 0);
  for (const l of links) {
    const s = typeof l.source === 'object' ? l.source.id : l.source;
    const t = typeof l.target === 'object' ? l.target.id : l.target;
    degreeMap.set(s, (degreeMap.get(s) ?? 0) + 1);
    degreeMap.set(t, (degreeMap.get(t) ?? 0) + 1);
  }
  return degreeMap;
}

export default function NetworkGraph({ data, selectedNode, onNodeClick, isDarkMode }: any) {
  const fgRef = useRef<any>(null);
  const [highlightNodes, setHighlightNodes] = useState(new Set<string>());
  const [highlightLinks, setHighlightLinks] = useState(new Set<any>());

  // Compute clusters and degrees
  const clusterMap = useMemo(() => computeClusters(data.nodes, data.links), [data]);
  const degreeMap = useMemo(() => computeDegrees(data.nodes, data.links), [data]);
  const maxDegree = useMemo(() => Math.max(1, ...degreeMap.values()), [degreeMap]);

  // Top N nodes by degree get labels
  const labelThreshold = useMemo(() => {
    const degrees = [...degreeMap.values()].sort((a, b) => b - a);
    const topN = Math.min(30, Math.floor(data.nodes.length * 0.05));
    return degrees[topN] ?? 1;
  }, [degreeMap, data.nodes.length]);

  // Highlight neighbors on selection
  useEffect(() => {
    const newNodes = new Set<string>();
    const newLinks = new Set<any>();

    if (selectedNode) {
      newNodes.add(selectedNode.id);
      data.links.forEach((link: any) => {
        const s = typeof link.source === 'object' ? link.source.id : link.source;
        const t = typeof link.target === 'object' ? link.target.id : link.target;
        if (s === selectedNode.id || t === selectedNode.id) {
          newLinks.add(link);
          newNodes.add(s);
          newNodes.add(t);
        }
      });
    }

    setHighlightNodes(newNodes);
    setHighlightLinks(newLinks);
  }, [selectedNode, data]);

  // Apply clustering force
  useEffect(() => {
    const fg = fgRef.current;
    if (!fg) return;

    // Cluster centroid force
    fg.d3Force('cluster', (alpha: number) => {
      const centroids = new Map<number, { x: number; y: number; count: number }>();

      // Compute centroids
      for (const node of data.nodes) {
        const c = clusterMap.get(node.id) ?? 0;
        if (!centroids.has(c)) centroids.set(c, { x: 0, y: 0, count: 0 });
        const centroid = centroids.get(c)!;
        centroid.x += node.x ?? 0;
        centroid.y += node.y ?? 0;
        centroid.count++;
      }

      for (const [, v] of centroids) {
        v.x /= v.count;
        v.y /= v.count;
      }

      // Pull nodes toward centroid
      const strength = 0.3 * alpha;
      for (const node of data.nodes) {
        const c = clusterMap.get(node.id) ?? 0;
        const centroid = centroids.get(c);
        if (centroid) {
          node.vx = (node.vx ?? 0) + (centroid.x - (node.x ?? 0)) * strength;
          node.vy = (node.vy ?? 0) + (centroid.y - (node.y ?? 0)) * strength;
        }
      }
    });

    // Strengthen charge to spread clusters apart
    fg.d3Force('charge')?.strength(-120);
    fg.d3Force('link')?.distance(30);

    fg.d3ReheatSimulation();
  }, [data, clusterMap]);

  const getNodeColor = useCallback((node: any) => {
    return RISK_NODE_COLORS[node.risk as keyof typeof RISK_NODE_COLORS] ?? RISK_NODE_COLORS.normal;
  }, []);

  const getNodeSize = useCallback((node: any) => {
    const degree = degreeMap.get(node.id) ?? 1;
    // Scale between 2 and 16 based on degree
    return 2 + (degree / maxDegree) * 14;
  }, [degreeMap, maxDegree]);

  const handleNodeClick = useCallback((node: any) => {
    if (!fgRef.current) return;

    if (selectedNode?.id !== node.id) {
      fgRef.current.centerAt(node.x, node.y, 800);
      fgRef.current.zoom(4, 800);
    }
    onNodeClick(node);
  }, [selectedNode, onNodeClick]);

  // Custom canvas node rendering
  const paintNode = useCallback((node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
    const isSelected = selectedNode?.id === node.id;
    const isHighlighted = highlightNodes.has(node.id);
    const isDimmed = selectedNode && !isHighlighted;
    const size = getNodeSize(node);
    const color = getNodeColor(node);
    const degree = degreeMap.get(node.id) ?? 0;

    // Glow for highlighted/selected nodes
    if ((isSelected || isHighlighted) && !isDimmed) {
      ctx.beginPath();
      ctx.arc(node.x, node.y, size * 2, 0, 2 * Math.PI);
      ctx.fillStyle = color + '30'; // 30 = ~19% opacity hex
      ctx.fill();
    }

    // Node circle
    ctx.beginPath();
    ctx.arc(node.x, node.y, size, 0, 2 * Math.PI);
    ctx.fillStyle = isDimmed ? (isDarkMode ? '#1e293b' : '#94a3b8') : color;
    ctx.globalAlpha = isDimmed ? 0.2 : 1;
    ctx.fill();

    // Border for selected
    if (isSelected) {
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2 / globalScale;
      ctx.stroke();
    }

    ctx.globalAlpha = 1;

    // Label for high-degree nodes
    if (degree >= labelThreshold && !isDimmed) {
      const label = node.id;
      const fontSize = Math.max(12 / globalScale, Math.min(size * 0.8, 16 / globalScale));
      ctx.font = `bold ${fontSize}px Sans-Serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';

      // Text shadow for readability
      ctx.strokeStyle = isDarkMode ? 'rgba(0,0,0,0.8)' : 'rgba(255,255,255,0.8)';
      ctx.lineWidth = 3 / globalScale;
      ctx.strokeText(label, node.x, node.y + size + fontSize * 0.7);

      ctx.fillStyle = isDarkMode ? '#ffffff' : '#000000';
      ctx.fillText(label, node.x, node.y + size + fontSize * 0.7);
    }
  }, [selectedNode, highlightNodes, getNodeSize, getNodeColor, degreeMap, labelThreshold, isDarkMode]);

  return (
    <div className="w-full h-full transition-colors duration-300 cursor-move">
      <ForceGraph2D
        ref={fgRef}
        graphData={data}
        backgroundColor={isDarkMode ? '#020617' : '#cbd5e1'}

        // Nodes
        nodeCanvasObject={paintNode}
        nodePointerAreaPaint={(node: any, color: string, ctx: CanvasRenderingContext2D) => {
          const size = getNodeSize(node);
          ctx.beginPath();
          ctx.arc(node.x, node.y, size, 0, 2 * Math.PI);
          ctx.fillStyle = color;
          ctx.fill();
        }}

        // Links
        linkColor={(link: any) => {
          if (highlightLinks.has(link)) return isDarkMode ? '#ffffff' : 'rgba(15,23,42,0.78)';
          const s = typeof link.source === 'object' ? link.source.id : link.source;
          const cluster = clusterMap.get(s) ?? 0;
          const clusterColor = CLUSTER_COLORS[cluster % CLUSTER_COLORS.length];
          return clusterColor + (isDarkMode ? '25' : '35');
        }}
        linkWidth={(link: any) => highlightLinks.has(link) ? 1.5 : 0.3}

        // Particles on highlighted links
        linkDirectionalParticles={(link: any) => highlightLinks.has(link) ? 3 : 0}
        linkDirectionalParticleWidth={2}
        linkDirectionalParticleSpeed={0.005}

        onNodeClick={handleNodeClick}
        cooldownTicks={200}
        warmupTicks={100}
      />
    </div>
  );
}
