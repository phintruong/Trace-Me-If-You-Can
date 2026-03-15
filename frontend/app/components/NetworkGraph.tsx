'use client';

import { useRef, useEffect, useMemo, useCallback, useState } from 'react';
import ForceGraph3D from 'react-force-graph-3d';
import ForceGraph2D from 'react-force-graph-2d';
import * as THREE from 'three';
import type { GraphData, GraphLink, GraphNode } from '../lib/api';

const RISK_NODE_COLORS = {
  normal: '#3772ff',
  suspicious: '#f97316',
  laundering: '#ef4444',
} as const;

const labelSpriteCache = new Map<string, THREE.SpriteMaterial>();

type SimulationNode = GraphNode & {
  x?: number;
  y?: number;
  z?: number;
  vx?: number;
  vy?: number;
  vz?: number;
  fx?: number;
  fy?: number;
};

type SimulationLink = Omit<GraphLink, 'source' | 'target'> & {
  source: string | SimulationNode;
  target: string | SimulationNode;
};

type ForceGraphHandle = {
  d3Force: (name: string, force?: ((alpha: number) => void)) => {
    strength?: (value: number) => void;
    distance?: (value: number) => void;
  } | undefined;
  d3ReheatSimulation: () => void;
  cameraPosition: (
    position: { x: number; y: number; z: number },
    lookAt?: SimulationNode,
    ms?: number
  ) => void;
};

type NetworkGraphProps = {
  data: GraphData;
  selectedNode: GraphNode | null;
  onNodeClick: (node: GraphNode) => void;
  isDarkMode: boolean;
  viewMode: '2d' | '3d';
};

function getLabelMaterial(text: string, isDarkMode: boolean) {
  const cacheKey = `${text}:${isDarkMode ? 'dark' : 'light'}`;
  const cached = labelSpriteCache.get(cacheKey);
  if (cached) return cached;

  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 160;
  const ctx = canvas.getContext('2d');

  if (ctx) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.font = 'bold 88px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.lineWidth = 12;
    ctx.strokeStyle = isDarkMode ? 'rgba(2, 6, 23, 0.9)' : 'rgba(255, 255, 255, 0.95)';
    ctx.fillStyle = isDarkMode ? '#f8fafc' : '#0f172a';
    ctx.strokeText(text, canvas.width / 2, canvas.height / 2);
    ctx.fillText(text, canvas.width / 2, canvas.height / 2);
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false });
  labelSpriteCache.set(cacheKey, material);
  return material;
}

function computeClusters(nodes: SimulationNode[], links: SimulationLink[]) {
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

function computeDegrees(nodes: SimulationNode[], links: SimulationLink[]) {
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

function cloneGraphData(data: GraphData): GraphData {
  return {
    nodes: data.nodes.map(n => ({ ...n })),
    links: data.links.map(l => ({ ...l })),
  };
}

export default function NetworkGraph({ data, selectedNode, onNodeClick, isDarkMode, viewMode }: NetworkGraphProps) {
  const DEFAULT_2D_ZOOM = 0.65;
  const DEFAULT_3D_DISTANCE = 420;
  const fgRef = useRef<ForceGraphHandle | null>(null);
  const fg2dRef = useRef<any>(null);
  const [hoverNode, setHoverNode] = useState<SimulationNode | null>(null);

  // Each mode gets its own deep-cloned data so d3 mutations don't cross-contaminate.
  const data3D = useMemo(() => cloneGraphData(data), [data]);
  const data2D = useMemo(() => cloneGraphData(data), [data]);

  const clusterMap = useMemo(() => computeClusters(data.nodes, data.links), [data]);
  const degreeMap = useMemo(() => computeDegrees(data.nodes, data.links), [data]);
  const maxDegree = useMemo(() => Math.max(1, ...degreeMap.values()), [degreeMap]);

  const topLaunderingNodeIds = useMemo(() => {
    return new Set(
      data.nodes
        .filter((node) => node.risk === 'laundering')
        .sort((a, b) => {
          const riskDelta = (b.riskScore ?? -Infinity) - (a.riskScore ?? -Infinity);
          if (riskDelta !== 0) return riskDelta;
          const txDelta = (b.txCount ?? 0) - (a.txCount ?? 0);
          if (txDelta !== 0) return txDelta;
          return (degreeMap.get(b.id) ?? 0) - (degreeMap.get(a.id) ?? 0);
        })
        .slice(0, 3)
        .map((node) => node.id)
    );
  }, [data.nodes, degreeMap]);

  const { highlightNodes, highlightLinks } = useMemo(() => {
    const newNodes = new Set<string>();
    const newLinks = new Set<SimulationLink>();

    if (selectedNode) {
      newNodes.add(selectedNode.id);
      (data.links as SimulationLink[]).forEach((link) => {
        const s = typeof link.source === 'object' ? link.source.id : link.source;
        const t = typeof link.target === 'object' ? link.target.id : link.target;
        if (s === selectedNode.id || t === selectedNode.id) {
          newLinks.add(link);
          newNodes.add(s);
          newNodes.add(t);
        }
      });
    }

    return { highlightNodes: newNodes, highlightLinks: newLinks };
  }, [selectedNode, data]);

  // --- Force configuration for 3D ---
  useEffect(() => {
    const fg = fgRef.current;
    if (!fg) return;

    fg.d3Force('cluster', (alpha: number) => {
      const centroids = new Map<number, { x: number; y: number; z: number; count: number }>();

      for (const node of data3D.nodes as SimulationNode[]) {
        const c = clusterMap.get(node.id) ?? 0;
        if (!centroids.has(c)) centroids.set(c, { x: 0, y: 0, z: 0, count: 0 });
        const centroid = centroids.get(c)!;
        centroid.x += node.x ?? 0;
        centroid.y += node.y ?? 0;
        centroid.z += node.z ?? 0;
        centroid.count++;
      }

      for (const [, value] of centroids) {
        value.x /= value.count;
        value.y /= value.count;
        value.z /= value.count;
      }

      const strength = 0.3 * alpha;
      for (const node of data3D.nodes as SimulationNode[]) {
        const c = clusterMap.get(node.id) ?? 0;
        const centroid = centroids.get(c);
        if (!centroid) continue;
        node.vx = (node.vx ?? 0) + (centroid.x - (node.x ?? 0)) * strength;
        node.vy = (node.vy ?? 0) + (centroid.y - (node.y ?? 0)) * strength;
        node.vz = (node.vz ?? 0) + (centroid.z - (node.z ?? 0)) * strength;
      }
    });

    fg.d3Force('charge')?.strength?.(-160);
    fg.d3Force('link')?.distance?.(45);
    fg.d3ReheatSimulation();
  }, [data3D, clusterMap]);

  // --- Force configuration for 2D ---
  useEffect(() => {
    const fg = fg2dRef.current;
    if (!fg) return;

    fg.d3Force('cluster', (alpha: number) => {
      const centroids = new Map<number, { x: number; y: number; count: number }>();

      for (const node of data2D.nodes as SimulationNode[]) {
        const c = clusterMap.get(node.id) ?? 0;
        if (!centroids.has(c)) centroids.set(c, { x: 0, y: 0, count: 0 });
        const centroid = centroids.get(c)!;
        centroid.x += node.x ?? 0;
        centroid.y += node.y ?? 0;
        centroid.count++;
      }

      for (const [, value] of centroids) {
        value.x /= value.count;
        value.y /= value.count;
      }

      const strength = 0.3 * alpha;
      for (const node of data2D.nodes as SimulationNode[]) {
        const c = clusterMap.get(node.id) ?? 0;
        const centroid = centroids.get(c);
        if (!centroid) continue;
        node.vx = (node.vx ?? 0) + (centroid.x - (node.x ?? 0)) * strength;
        node.vy = (node.vy ?? 0) + (centroid.y - (node.y ?? 0)) * strength;
      }
    });

    fg.d3Force('charge')?.strength?.(-200);
    fg.d3Force('link')?.distance?.(60);
    fg.d3ReheatSimulation();
  }, [data2D, clusterMap]);

  const getNodeColor = useCallback((node: GraphNode) => {
    return RISK_NODE_COLORS[node.risk as keyof typeof RISK_NODE_COLORS] ?? RISK_NODE_COLORS.normal;
  }, []);

  const getNodeSize = useCallback((node: GraphNode) => {
    const degree = degreeMap.get(node.id) ?? 1;
    return 2.5 + (degree / maxDegree) * 12;
  }, [degreeMap, maxDegree]);

  const handleNodeClick = useCallback((node: SimulationNode) => {
    if (viewMode === '3d') {
      if (!fgRef.current) return;

      if (selectedNode?.id !== node.id) {
        const nodeX = node.x ?? 0;
        const nodeY = node.y ?? 0;
        const nodeZ = node.z ?? 0;

        // Third-person angled view from above — offset behind and elevated
        const offsetBack = 200;
        const offsetUp = 250;
        const offsetSide = 120;

        fgRef.current.cameraPosition(
          {
            x: nodeX + offsetSide,
            y: nodeY + offsetUp,
            z: nodeZ + offsetBack,
          },
          node,
          1000
        );
      }
    } else {
      if (fg2dRef.current && selectedNode?.id !== node.id) {
        fg2dRef.current.centerAt(node.x, node.y, 1000);
        fg2dRef.current.zoom(4, 1000);
      }
    }

    onNodeClick(node);
  }, [selectedNode, onNodeClick, viewMode]);

  useEffect(() => {
    if (viewMode === '3d') {
      const fg = fgRef.current;
      if (!selectedNode) {
        fg?.cameraPosition({ x: 0, y: 0, z: DEFAULT_3D_DISTANCE }, undefined, 1000);
        return;
      }

      const node = (data3D.nodes as SimulationNode[]).find((candidate) => candidate.id === selectedNode.id);
      if (!fg || !node) return;

      const distance = 120;
      const distRatio = 1 + distance / Math.max(1, Math.hypot(node.x ?? 0, node.y ?? 0, node.z ?? 0));

      fg.cameraPosition(
        {
          x: (node.x ?? 0) * distRatio,
          y: (node.y ?? 0) * distRatio,
          z: (node.z ?? 0) * distRatio,
        },
        node,
        1000
      );
      return;
    }

    const fg = fg2dRef.current;
    if (!selectedNode) {
      fg?.centerAt(0, 0, 1000);
      fg?.zoom(DEFAULT_2D_ZOOM, 1000);
      return;
    }

    const node = (data2D.nodes as SimulationNode[]).find((candidate) => candidate.id === selectedNode.id);
    if (!fg || !node) return;

    fg.centerAt(node.x ?? 0, node.y ?? 0, 1000);
    fg.zoom(4, 1000);
  }, [selectedNode, viewMode, data3D, data2D, DEFAULT_2D_ZOOM, DEFAULT_3D_DISTANCE]);

  // --- 2D Canvas Rendering ---
  const paintNode2D = useCallback((node: SimulationNode, ctx: CanvasRenderingContext2D) => {
    const isSelected = selectedNode?.id === node.id;
    const isHovered = hoverNode?.id === node.id;
    const isHighlighted = highlightNodes.has(node.id);
    const isDimmed = Boolean(selectedNode) && !isHighlighted;
    const size = getNodeSize(node) * 0.5;
    const color = getNodeColor(node);
    const x = node.x ?? 0;
    const y = node.y ?? 0;

    ctx.save();

    // Outer glow via radial gradient
    if ((isSelected || isHovered || isHighlighted) && !isDimmed) {
      const glowRadius = size * (isSelected ? 3.5 : isHovered ? 3 : 2.2);
      const gradient = ctx.createRadialGradient(x, y, size * 0.5, x, y, glowRadius);
      gradient.addColorStop(0, color);
      gradient.addColorStop(0.4, color + '44');
      gradient.addColorStop(1, color + '00');
      ctx.beginPath();
      ctx.arc(x, y, glowRadius, 0, 2 * Math.PI);
      ctx.fillStyle = gradient;
      ctx.globalAlpha = isSelected ? 0.6 : isHovered ? 0.5 : 0.3;
      ctx.fill();
      ctx.globalAlpha = 1;
    }

    // Node body
    const drawSize = isSelected ? size * 1.3 : isHovered ? size * 1.15 : size;
    ctx.beginPath();
    ctx.arc(x, y, drawSize, 0, 2 * Math.PI);

    if (isDimmed) {
      ctx.fillStyle = isDarkMode ? '#1e293b' : '#94a3b8';
      ctx.globalAlpha = 0.15;
    } else {
      const innerGrad = ctx.createRadialGradient(x - drawSize * 0.3, y - drawSize * 0.3, 0, x, y, drawSize);
      innerGrad.addColorStop(0, '#ffffff88');
      innerGrad.addColorStop(0.3, color);
      innerGrad.addColorStop(1, color + 'cc');
      ctx.fillStyle = innerGrad;
      ctx.globalAlpha = 1;
    }
    ctx.fill();

    // Border ring
    if (!isDimmed) {
      ctx.strokeStyle = isSelected ? '#ffffff' : color;
      ctx.lineWidth = isSelected ? 0.8 : 0.3;
      ctx.globalAlpha = isSelected ? 0.9 : 0.5;
      ctx.stroke();
    }

    ctx.globalAlpha = 1;

    // Label for top laundering nodes or hovered/selected
    const showLabel = topLaunderingNodeIds.has(node.id) || isSelected || isHovered;
    if (showLabel && !isDimmed) {
      const label = node.id;
      ctx.font = `bold ${isSelected || isHovered ? 5 : 4}px sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      const labelY = y - drawSize - 4;

      ctx.strokeStyle = isDarkMode ? 'rgba(2, 6, 23, 0.95)' : 'rgba(255, 255, 255, 0.95)';
      ctx.lineWidth = 2;
      ctx.lineJoin = 'round';
      ctx.strokeText(label, x, labelY);

      ctx.fillStyle = isDarkMode ? '#f8fafc' : '#0f172a';
      ctx.fillText(label, x, labelY);
    }

    ctx.restore();
  }, [selectedNode, hoverNode, highlightNodes, getNodeSize, getNodeColor, isDarkMode, topLaunderingNodeIds]);

  // --- 3D node rendering ---
  const renderNode = useCallback((node: SimulationNode) => {
    const isSelected = selectedNode?.id === node.id;
    const isHighlighted = highlightNodes.has(node.id);
    const isDimmed = Boolean(selectedNode) && !isHighlighted;
    const size = getNodeSize(node);
    const color = getNodeColor(node);

    const group = new THREE.Group();

    const sphere = new THREE.Mesh(
      new THREE.SphereGeometry(size, 18, 18),
      new THREE.MeshStandardMaterial({
        color: isDimmed ? (isDarkMode ? '#1e293b' : '#94a3b8') : color,
        transparent: true,
        opacity: isDimmed ? 0.2 : 1,
        emissive: isDimmed ? '#000000' : color,
        emissiveIntensity: isSelected ? 0.75 : isHighlighted ? 0.35 : 0.18,
        roughness: 0.35,
        metalness: 0.1,
      })
    );
    group.add(sphere);

    if ((isSelected || isHighlighted) && !isDimmed) {
      const aura = new THREE.Mesh(
        new THREE.SphereGeometry(size * 1.8, 16, 16),
        new THREE.MeshBasicMaterial({
          color,
          transparent: true,
          opacity: isSelected ? 0.16 : 0.08,
        })
      );
      group.add(aura);
    }

    if (topLaunderingNodeIds.has(node.id) && !isDimmed) {
      const label = new THREE.Sprite(getLabelMaterial(node.id, isDarkMode));
      label.position.set(0, size + 8, 0);
      label.scale.set(32, 10, 1);
      group.add(label);
    }

    return group;
  }, [selectedNode, highlightNodes, getNodeSize, getNodeColor, isDarkMode, topLaunderingNodeIds]);

  // --- Shared link callbacks ---
  const linkColorFn = useCallback(() => (
    isDarkMode ? 'rgba(247,243,197,0.35)' : 'rgba(50,32,53,0.35)'
  ), [isDarkMode]);

  const linkWidthFn = useCallback(() => 0.8, []);

  const linkParticlesFn = useCallback(() => 0, []);

  const nodeLabelFn = useCallback((node: SimulationNode) => topLaunderingNodeIds.has(node.id) ? node.id : '', [topLaunderingNodeIds]);

  const bgColor = isDarkMode ? '#020617' : '#FFFEEF';

  // Render BOTH graphs, hide inactive one with CSS to avoid destroy/recreate tick errors
  return (
    <div className="w-full h-full transition-colors duration-300 cursor-move">
      <div style={{ width: '100%', height: '100%', display: viewMode === '3d' ? 'block' : 'none' }}>
        <ForceGraph3D
          ref={fgRef as any}
          graphData={data3D as any}
          backgroundColor={bgColor}
          showNavInfo={false}
          nodeThreeObject={renderNode}
          nodeLabel={nodeLabelFn as any}
          linkColor={linkColorFn as any}
          linkOpacity={0.45}
          linkWidth={linkWidthFn as any}
          linkDirectionalParticles={linkParticlesFn as any}
          linkDirectionalParticleWidth={2}
          linkDirectionalParticleSpeed={0.005}
          onNodeClick={handleNodeClick as any}
          enableNodeDrag={false}
          cooldownTicks={200}
          warmupTicks={100}
        />
      </div>
      <div style={{ width: '100%', height: '100%', display: viewMode === '2d' ? 'block' : 'none' }}>
        <ForceGraph2D
          ref={fg2dRef as any}
          graphData={data2D as any}
          backgroundColor={bgColor}
          nodeCanvasObject={paintNode2D as any}
          nodePointerAreaPaint={(node: SimulationNode, color: string, ctx: CanvasRenderingContext2D) => {
            const size = getNodeSize(node) * 0.5;
            ctx.beginPath();
            ctx.arc(node.x ?? 0, node.y ?? 0, size + 2, 0, 2 * Math.PI);
            ctx.fillStyle = color;
            ctx.fill();
          }}
          onNodeHover={(node: SimulationNode | null) => setHoverNode(node)}
          nodeLabel={() => ''}
          linkColor={linkColorFn as any}
          linkWidth={linkWidthFn as any}
          linkDirectionalParticles={linkParticlesFn as any}
          linkDirectionalParticleWidth={2}
          linkDirectionalParticleSpeed={0.005}
          onNodeClick={handleNodeClick as any}
          onNodeDragEnd={(node: SimulationNode) => {
            node.fx = node.x;
            node.fy = node.y;
          }}
          cooldownTicks={200}
          warmupTicks={100}
        />
      </div>
    </div>
  );
}
