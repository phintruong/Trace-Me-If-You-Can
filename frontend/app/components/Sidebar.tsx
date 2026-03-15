import { type PointerEvent, useEffect, useState } from 'react';
import { AlertTriangle, ShieldCheck, Activity, ArrowRight, ArrowLeft } from 'lucide-react';

export default function Sidebar({ 
  selectedAccount, 
  onClose, 
  allLinks = [],
  onResizeStart,
  isDarkMode
}: { 
  selectedAccount: any, 
  onClose: () => void,
  allLinks?: any[],
  onResizeStart: (event: PointerEvent<HTMLDivElement>) => void,
  isDarkMode: boolean
}) {
  const [counterpartyFilter, setCounterpartyFilter] = useState('');
  const isHighRisk = selectedAccount?.risk === 'laundering';
  const isMediumRisk = selectedAccount?.risk === 'suspicious';
  const panelClasses = isDarkMode
    ? 'border-slate-700 bg-slate-900 text-slate-200'
    : 'border-slate-500 bg-slate-300 text-slate-900';
  const titleClasses = isDarkMode ? 'text-white' : 'text-slate-950';
  const mutedTextClasses = isDarkMode ? 'text-slate-400' : 'text-slate-700';
  const bodyTextClasses = isDarkMode ? 'text-slate-300' : 'text-slate-800';
  const surfaceClasses = isDarkMode
    ? 'border-slate-700 bg-slate-800'
    : 'border-slate-400 bg-slate-100';
  const resizeHandleClasses = isDarkMode
    ? 'bg-slate-700 hover:bg-slate-500'
    : 'bg-slate-500 hover:bg-slate-700';

  const accountTransactions = allLinks.filter(link => {
    const sourceId = link.source.id || link.source;
    const targetId = link.target.id || link.target;
    return sourceId === selectedAccount?.id || targetId === selectedAccount?.id;
  });

  const filteredTransactions = !counterpartyFilter.trim()
    ? accountTransactions
    : accountTransactions.filter(link => {
        const sourceId = link.source.id || link.source;
        const targetId = link.target.id || link.target;
        const counterpartId = sourceId === selectedAccount?.id ? targetId : sourceId;
        return counterpartId.toLowerCase().includes(counterpartyFilter.trim().toLowerCase());
      });

  useEffect(() => {
    setCounterpartyFilter('');
  }, [selectedAccount?.id]);

  return (
    <div className={`relative h-full w-full overflow-hidden border-l p-6 shadow-2xl transition-colors duration-300 ${panelClasses}`}>
      <div
        role="separator"
        aria-label="Resize sidebar"
        aria-orientation="vertical"
        onPointerDown={onResizeStart}
        className="absolute left-0 top-0 h-full w-3 -translate-x-1/2 cursor-col-resize touch-none"
      >
        <div className={`absolute inset-y-0 left-1/2 w-px -translate-x-1/2 transition-colors ${resizeHandleClasses}`} />
      </div>

      <div className="flex justify-between items-center shrink-0">
        <h2 className={`text-xl font-bold tracking-tight ${titleClasses}`}>Account Details</h2>
        <button
          onClick={onClose}
          className={`rounded-md p-2 text-4xl leading-none transition-colors ${isDarkMode ? 'text-slate-400 hover:text-white hover:bg-slate-800' : 'text-slate-700 hover:text-slate-950 hover:bg-slate-200'}`}
          aria-label="Close sidebar"
        >
          &times;
        </button>
      </div>

      {selectedAccount && (
        <div className="flex flex-col h-full overflow-hidden space-y-6">
          
          <div className="space-y-6 shrink-0">
            <div>
              <p className={`mb-1 text-sm uppercase tracking-wider ${mutedTextClasses}`}>Account ID</p>
              <p className={`text-2xl font-mono ${titleClasses}`}>{selectedAccount.id}</p>
              
              {/* Rest of your risk badge code remains exactly the same */}
              <div className={`mt-3 inline-flex items-center gap-2 px-3 py-1 rounded-full text-sm font-semibold
                ${isHighRisk ? 'bg-red-100 text-red-700 border border-red-200 dark:bg-red-500/20 dark:text-red-400 dark:border-red-500/50' : 
                  isMediumRisk ? 'bg-orange-100 text-orange-700 border border-orange-200 dark:bg-orange-500/20 dark:text-orange-400 dark:border-orange-500/50' : 
                  'bg-blue-100 text-blue-700 border border-blue-200 dark:bg-blue-500/20 dark:text-blue-400 dark:border-blue-500/50'}`}>
                {isHighRisk && <AlertTriangle size={16} />}
                {isMediumRisk && <Activity size={16} />}
                {!isHighRisk && !isMediumRisk && <ShieldCheck size={16} />}
                {selectedAccount.risk.toUpperCase()}
              </div>
            </div>

            <div>
              <h3 className={`mb-2 text-sm uppercase tracking-wider ${mutedTextClasses}`}>Pattern Detected</h3>
              <p className={`font-semibold ${isDarkMode ? 'text-slate-100' : 'text-slate-900'}`}>{selectedAccount.pattern}</p>
            </div>

            <div>
              <h3 className={`mb-2 text-sm uppercase tracking-wider ${mutedTextClasses}`}>AI Explanation</h3>
              <p className={`text-sm leading-relaxed ${bodyTextClasses}`}>
                {selectedAccount.aiExplanation}
              </p>
            </div>
          </div>

          {/* Transactions List remains exactly the same */}
          <div className={`flex flex-col flex-1 min-h-0 border-t pt-6 transition-colors ${isDarkMode ? 'border-slate-700' : 'border-slate-200'}`}>
            <div className="mb-4 shrink-0">
              <label
                htmlFor="counterparty-filter"
                className={`mb-2 block text-sm uppercase tracking-wider ${mutedTextClasses}`}
              >
                Filter User
              </label>
              <input
                id="counterparty-filter"
                type="text"
                value={counterpartyFilter}
                onChange={(e) => setCounterpartyFilter(e.target.value)}
                placeholder="Type an account ID"
                className={`w-full rounded-lg border px-3 py-2 text-sm outline-none transition-colors ${
                  isDarkMode
                    ? 'border-slate-700 bg-slate-800 text-white placeholder-slate-500 focus:border-slate-500'
                    : 'border-slate-500 bg-slate-100 text-slate-950 placeholder-slate-600 focus:border-slate-700'
                }`}
              />
            </div>

            <div className="flex justify-between items-end mb-3 shrink-0">
              <h3 className={`text-sm uppercase tracking-wider ${mutedTextClasses}`}>Known Transactions</h3>
              <span className={`text-xs ${mutedTextClasses}`}>{filteredTransactions.length} records</span>
            </div>
             
            <div className="flex-1 overflow-y-auto pr-2 space-y-2 pb-4 scrollbar-thin scrollbar-thumb-slate-300 dark:scrollbar-thumb-slate-700 scrollbar-track-transparent">
              {filteredTransactions.length === 0 ? (
                <p className={`text-sm italic ${mutedTextClasses}`}>
                  {accountTransactions.length === 0 ? 'No connections found.' : 'No transactions match the selected user.'}
                </p>
              ) : (
                filteredTransactions.map((tx, idx) => {
                  const sourceId = tx.source.id || tx.source;
                  const targetId = tx.target.id || tx.target;
                  const isOutgoing = sourceId === selectedAccount.id;
                  const counterpartId = isOutgoing ? targetId : sourceId;

                  return (
                    <div
                      key={idx}
                      className={`flex flex-col gap-1 rounded-lg border p-3 transition-colors ${
                        isDarkMode
                          ? `${surfaceClasses} hover:border-slate-500`
                          : `${surfaceClasses} hover:border-slate-600`
                      }`}
                    >
                      <div className="flex justify-between items-center">
                        <div className={`flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider ${isOutgoing ? 'text-orange-500 dark:text-orange-400' : 'text-emerald-600 dark:text-emerald-400'}`}>
                          {isOutgoing ? <ArrowRight size={14} /> : <ArrowLeft size={14} />}
                          {isOutgoing ? 'Sent to' : 'Received from'}
                        </div>
                        <div className={`font-mono text-sm font-semibold ${isDarkMode ? 'text-slate-200' : 'text-slate-950'}`}>
                          ${tx.amount.toLocaleString()}
                        </div>
                      </div>
                      <div className={`ml-5 font-mono text-sm ${mutedTextClasses}`}>
                        {counterpartId}
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>

        </div>
      )}
    </div>
  );
}
