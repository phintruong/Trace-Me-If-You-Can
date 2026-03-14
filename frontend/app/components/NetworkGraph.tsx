'use client';

import { useRef, useEffect, useState } from 'react';
import ForceGraph3D from 'react-force-graph-3d';
import * as THREE from 'three';

// --- Helper: Generate Glowing Textures ---
// We cache the textures so we don't murder the GPU by creating thousands of canvases
const textureCache: Record<string, THREE.CanvasTexture> = {};

function getGlowTexture(color: string) {
  if (textureCache[color]) return textureCache[color];

  const canvas = document.createElement('canvas');
  canvas.width = 64;
  canvas.height = 64;
  const ctx = canvas.getContext('2d');

  if (ctx) {
    const gradient = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
    gradient.addColorStop(0, 'rgba(255, 255, 255, 1)'); // Hot white center
    gradient.addColorStop(0.3, color); // Core color
    gradient.addColorStop(1, 'rgba(0, 0, 0, 0)'); // Fade to transparent

    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 64, 64);
  }

  const texture = new THREE.CanvasTexture(canvas);
  textureCache[color] = texture;
  return texture;
}

export default function NetworkGraph({ data, selectedNode, onNodeClick, isDarkMode }: any) {
  const fgRef = useRef<any>();
  const [highlightNodes, setHighlightNodes] = useState(new Set());
  const [highlightLinks, setHighlightLinks] = useState(new Set());

  useEffect(() => {
    const newHighlightNodes = new Set();
    const newHighlightLinks = new Set();

    if (selectedNode) {
      newHighlightNodes.add(selectedNode.id);
      data.links.forEach((link: any) => {
        const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
        const targetId = typeof link.target === 'object' ? link.target.id : link.target;

        if (sourceId === selectedNode.id || targetId === selectedNode.id) {
          newHighlightLinks.add(link);
          newHighlightNodes.add(sourceId);
          newHighlightNodes.add(targetId);
        }
      });
    }

    setHighlightNodes(newHighlightNodes);
    setHighlightLinks(newHighlightLinks);
  }, [selectedNode, data]);

  const getBaseColor = (node: any) => {
    if (node.risk === 'laundering') return '#ef4444'; // Red
    if (node.risk === 'suspicious') return '#f97316'; // Orange
    return isDarkMode ? '#38bdf8' : '#3b82f6'; // Light Blue / Standard Blue
  };

  const handleNodeClick = (node: any) => {
    if (!fgRef.current) return;

    if (selectedNode?.id !== node.id) {
      const distance = 80; 
      const distRatio = 1 + distance / Math.hypot(node.x, node.y, node.z);

      fgRef.current.cameraPosition(
        { x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio }, 
        node, 
        1500  
      );
    }
    onNodeClick(node);
  };

  return (
    <div className="w-full h-full transition-colors duration-300 cursor-move">
      <ForceGraph3D
        ref={fgRef}
        graphData={data}
        
        // --- 3D Custom Glowing Nodes ---
        nodeThreeObject={(node: any) => {
          const isSelected = selectedNode?.id === node.id;
          const isHighlighted = highlightNodes.has(node.id);
          const isDimmed = selectedNode && !isHighlighted;
          const baseColor = getBaseColor(node);

          // Create a group to hold both the solid sphere and the glow
          const group = new THREE.Group();

          // 1. The Solid Core
          const coreSize = isSelected ? 8 : (isHighlighted ? 5 : 3);
          const geometry = new THREE.SphereGeometry(coreSize, 16, 16);
          const material = new THREE.MeshPhongMaterial({
            color: isDimmed ? (isDarkMode ? '#1e293b' : '#cbd5e1') : baseColor,
            transparent: true,
            opacity: isDimmed ? 0.3 : 1,
            shininess: 100 // Makes it look a bit like glass/plastic
          });
          const sphere = new THREE.Mesh(geometry, material);
          group.add(sphere);

          // 2. The Glowing Aura (Only if not dimmed)
          if (!isDimmed) {
            const glowMaterial = new THREE.SpriteMaterial({
              map: getGlowTexture(baseColor),
              color: baseColor,
              transparent: true,
              blending: THREE.AdditiveBlending, // This makes the light compound and "glow"
              depthWrite: false, // Prevents weird clipping issues with other transparent objects
              opacity: isSelected ? 0.8 : 0.4 // Pulse intensity based on selection
            });
            const sprite = new THREE.Sprite(glowMaterial);
            
            // Scale the glow to be much larger than the core
            const glowScale = isSelected ? 35 : (isHighlighted ? 20 : 12);
            sprite.scale.set(glowScale, glowScale, 1);
            group.add(sprite);
          }

          return group;
        }}

        // --- Environment ---
        backgroundColor={isDarkMode ? '#020617' : '#f8fafc'}
        showNavInfo={false} 
        
        // --- Links & Particles ---
        linkColor={(link: any) => highlightLinks.has(link)
          ? (isDarkMode ? '#ffffff' : 'rgba(0, 0, 0, 0.7)')
          : (isDarkMode ? 'rgba(255, 255, 255, 0.2)' : 'rgba(0, 0, 0, 0.2)')
        }
        linkOpacity={0.5}
        linkWidth={(link: any) => highlightLinks.has(link) ? 1.5 : 0.2}
        
        linkDirectionalParticles={(link: any) => highlightLinks.has(link) ? 3 : 1}
        linkDirectionalParticleWidth={(link: any) => highlightLinks.has(link) ? 2 : 1}
        linkDirectionalParticleSpeed={0.005}
        linkDirectionalParticleColor={(link: any) => {
          const sourceNodeId = typeof link.source === 'object' ? link.source.id : link.source;
          const sourceNode = data.nodes.find((n: any) => n.id === sourceNodeId);
          return sourceNode ? getBaseColor(sourceNode) : '#94a3b8';
        }}

        onNodeClick={handleNodeClick}
      />
    </div>
  );
}
