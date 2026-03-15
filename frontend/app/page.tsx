'use client';

import { type PointerEvent as ReactPointerEvent, useCallback, useEffect, useState } from 'react';
import dynamic from 'next/dynamic';
import Sidebar from './components/Sidebar';
import { type GraphData, runPipeline, fetchGraph } from './lib/api';
import { Moon, Sun, Search, Loader2 } from 'lucide-react';

const NetworkGraph = dynamic(() => import('./components/NetworkGraph'), { ssr: false });
const SIDEBAR_MIN_WIDTH = 320;
const SIDEBAR_MAX_WIDTH = 640;

export default function Dashboard() {
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], links: [] });
  const [selectedAccount, setSelectedAccount] = useState<any>(null);
  const [isDarkMode, setIsDarkMode] = useState(true);
  const [globalSearch, setGlobalSearch] = useState('');
  const [sidebarWidth, setSidebarWidth] = useState(384);
  const [isResizingSidebar, setIsResizingSidebar] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pipelineRunning, setPipelineRunning] = useState(false);

  // Load data: try cached graph first, show message if no data yet
  const loadData = useCallback(async () => {
    if (pipelineRunning) return; // prevent duplicate calls
    setLoading(true);
    setError(null);
    try {
      const data = await fetchGraph();
      if (data.nodes.length > 0) {
        setGraphData(data);
      } else {
        setError('No data yet. Click "Run Pipeline" to analyze transactions.');
      }
    } catch {
      setError('No data yet. Click "Run Pipeline" to analyze transactions.');
    } finally {
      setLoading(false);
    }
  }, [pipelineRunning]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleNodeClick = (node: any) => {
    setSelectedAccount((prev: any) => (prev?.id === node.id ? null : node));
  };

  const executeSearch = (term: string) => {
    if (!term.trim()) return;
    const foundNode = graphData.nodes.find(n => n.id.toLowerCase() === term.toLowerCase());

    if (foundNode) {
      setSelectedAccount(foundNode);
      setGlobalSearch('');
    } else {
      alert(`Account "${term}" not found in current network.`);
    }
  };

  useEffect(() => {
    if (!isResizingSidebar) return;

    const handlePointerMove = (event: globalThis.PointerEvent) => {
      const nextWidth = window.innerWidth - event.clientX;
      const clampedWidth = Math.min(SIDEBAR_MAX_WIDTH, Math.max(SIDEBAR_MIN_WIDTH, nextWidth));
      setSidebarWidth(clampedWidth);
    };

    const stopResizing = () => {
      setIsResizingSidebar(false);
    };

    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', stopResizing);
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';

    return () => {
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', stopResizing);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
  }, [isResizingSidebar]);

  return (
    <div className={`relative h-screen w-screen overflow-hidden ${isDarkMode ? 'dark bg-slate-950' : 'bg-slate-50'}`}>
      {/* Top Bar */}
      <div className={`absolute top-0 left-0 right-0 z-10 flex items-center justify-between px-6 py-3 backdrop-blur-md ${isDarkMode ? 'bg-slate-900/80 border-b border-slate-700' : 'bg-white/80 border-b border-slate-200'}`}>
        <h1 className={`text-lg font-bold tracking-tight ${isDarkMode ? 'text-white' : 'text-slate-900'}`}>
          GenAI Genesis — AML Network
        </h1>

        <div className="flex items-center gap-3">
          {/* Run Pipeline */}
          <button
            onClick={async () => {
              setPipelineRunning(true);
              setError(null);
              try {
                const data = await runPipeline();
                setGraphData(data);
                setSelectedAccount(null);
              } catch (err: any) {
                setError(err.message);
              } finally {
                setPipelineRunning(false);
              }
            }}
            disabled={pipelineRunning}
            className={`flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-sm font-medium transition-colors ${
              isDarkMode
                ? 'border-slate-700 text-slate-300 hover:bg-slate-800 disabled:opacity-50'
                : 'border-slate-200 text-slate-700 hover:bg-slate-100 disabled:opacity-50'
            }`}
          >
            {pipelineRunning && <Loader2 size={14} className="animate-spin" />}
            {pipelineRunning ? 'Running...' : 'Run Pipeline'}
          </button>

          {/* Search */}
          <div className="relative">
            <Search className={`absolute left-3 top-1/2 -translate-y-1/2 ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`} size={16} />
            <input
              type="text"
              value={globalSearch}
              onChange={(e) => setGlobalSearch(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && executeSearch(globalSearch)}
              placeholder="Search account..."
              className={`rounded-lg border pl-9 pr-3 py-1.5 text-sm outline-none transition-colors ${
                isDarkMode
                  ? 'border-slate-700 bg-slate-800 text-white placeholder-slate-500 focus:border-slate-500'
                  : 'border-slate-200 bg-slate-50 text-slate-900 placeholder-slate-400 focus:border-slate-400'
              }`}
            />
          </div>

          {/* Dark mode toggle */}
          <button
            onClick={() => setIsDarkMode(!isDarkMode)}
            className={`rounded-lg p-2 transition-colors ${isDarkMode ? 'text-slate-400 hover:text-white hover:bg-slate-800' : 'text-slate-500 hover:text-slate-900 hover:bg-slate-200'}`}
          >
            {isDarkMode ? <Sun size={18} /> : <Moon size={18} />}
          </button>
        </div>
      </div>

      {/* 3D Graph */}
      <div className="h-full w-full" style={{ paddingRight: selectedAccount ? sidebarWidth : 0 }}>
        {loading ? (
          <div className="flex h-full items-center justify-center">
            <div className="flex flex-col items-center gap-3">
              <Loader2 size={32} className={`animate-spin ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`} />
              <p className={`text-sm ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>
                {pipelineRunning ? 'Running pipeline (this may take a minute)...' : 'Loading graph data...'}
              </p>
            </div>
          </div>
        ) : error ? (
          <div className="flex h-full items-center justify-center">
            <div className="flex flex-col items-center gap-3">
              <p className="text-sm text-red-500">{error}</p>
              <button onClick={loadData} className="rounded-lg border border-red-500 px-4 py-1.5 text-sm text-red-500 hover:bg-red-500/10">
                Retry
              </button>
            </div>
          </div>
        ) : (
          <NetworkGraph
            data={graphData}
            selectedNode={selectedAccount}
            onNodeClick={handleNodeClick}
            isDarkMode={isDarkMode}
          />
        )}
      </div>

      {/* Sidebar */}
      {selectedAccount && (
        <div className="absolute top-0 right-0 h-full" style={{ width: sidebarWidth }}>
          <Sidebar
            selectedAccount={selectedAccount}
            onClose={() => setSelectedAccount(null)}
            allLinks={graphData.links}
            onResizeStart={(e: ReactPointerEvent<HTMLDivElement>) => {
              e.preventDefault();
              setIsResizingSidebar(true);
            }}
            isDarkMode={isDarkMode}
          />
        </div>
      )}
    </div>
  );
}
