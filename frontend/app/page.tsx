'use client';

import { type PointerEvent as ReactPointerEvent, useEffect, useState } from 'react';
import dynamic from 'next/dynamic';
import Sidebar from './components/Sidebar';
import { graphData } from './lib/mockData';
import { Moon, Sun, Search } from 'lucide-react';

const NetworkGraph = dynamic(() => import('./components/NetworkGraph'), { ssr: false });
const SIDEBAR_MIN_WIDTH = 320;
const SIDEBAR_MAX_WIDTH = 640;

export default function Dashboard() {
  const [selectedAccount, setSelectedAccount] = useState<any>(null);
  const [isDarkMode, setIsDarkMode] = useState(true);
  const [globalSearch, setGlobalSearch] = useState('');
  const [sidebarWidth, setSidebarWidth] = useState(384);
  const [isResizingSidebar, setIsResizingSidebar] = useState(false);

  const handleNodeClick = (node: any) => {
    setSelectedAccount((prev: any) => (prev?.id === node.id ? null : node));
  };

  const executeSearch = (term: string) => {
    if (!term.trim()) return;
    const foundNode = graphData.nodes.find(n => n.id.toLowerCase() === term.toLowerCase());
    
    if (foundNode) {
      setSelectedAccount(foundNode);
      setGlobalSearch(''); // Clear input on success
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
    <main className={`flex h-screen w-screen overflow-hidden font-sans transition-colors duration-300 ${isDarkMode ? 'dark bg-slate-950' : 'bg-slate-50'}`}>
      
      {/* Main Graph Area */}
      <div className="flex-1 relative">
        
        {/* Header Overlay */}
        <div className="absolute top-0 left-0 p-6 z-10 pointer-events-none">
          <h1 className={`text-2xl font-bold tracking-tight drop-shadow-md transition-colors ${isDarkMode ? 'text-white' : 'text-slate-900'}`}>
            FinTrace <span className={`font-light ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>Investigation Tool</span>
          </h1>
          <p className={`text-sm mt-1 transition-colors ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>
            Select a node or search to inspect transaction vectors.
          </p>
        </div>

        {/* Global Search Bar (Top Center) */}
        <div className={`absolute top-6 left-1/2 -translate-x-1/2 z-20 flex items-center rounded-full px-4 py-2 shadow-lg transition-colors ${
          isDarkMode
            ? 'bg-slate-900 border border-slate-700'
            : 'bg-white border border-slate-300'
        }`}>
          <Search
            size={18}
            className={`mr-2 transition-colors ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}
          />
          <input 
            type="text" 
            placeholder="Search Account ID (e.g. ACC-1002)" 
            value={globalSearch}
            onChange={(e) => setGlobalSearch(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && executeSearch(globalSearch)}
            className={`w-64 bg-transparent border-none outline-none text-sm transition-colors ${
              isDarkMode
                ? 'text-white placeholder-slate-500'
                : 'text-slate-900 placeholder-slate-400'
            }`}
          />
        </div>

        {/* Theme Toggle Button */}
        <button 
          onClick={() => setIsDarkMode(!isDarkMode)}
          className={`absolute top-6 right-6 z-20 p-2 rounded-full border transition-colors shadow-md ${
            isDarkMode 
              ? 'bg-slate-800 border-slate-700 text-yellow-400 hover:bg-slate-700' 
              : 'bg-white border-slate-300 text-slate-700 hover:bg-slate-100'
          }`}
        >
          {isDarkMode ? <Sun size={20} /> : <Moon size={20} />}
        </button>

        {/* The Graph */}
        <NetworkGraph 
          data={graphData} 
          selectedNode={selectedAccount} 
          onNodeClick={handleNodeClick} 
          isDarkMode={isDarkMode} 
        />
      </div>

      {selectedAccount && (
        <div
          className="absolute top-0 right-0 h-full z-50 transition-transform duration-300 ease-in-out"
          style={{ width: `${sidebarWidth}px` }}
        >
          <Sidebar
            selectedAccount={selectedAccount}
            onClose={() => setSelectedAccount(null)}
            allLinks={graphData.links}
            isDarkMode={isDarkMode}
            onResizeStart={(event: ReactPointerEvent<HTMLDivElement>) => {
              event.preventDefault();
              setIsResizingSidebar(true);
            }}
          />
        </div>
      )}

    </main>
  );
}
