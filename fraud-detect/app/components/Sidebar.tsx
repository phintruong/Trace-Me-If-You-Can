import { type PointerEvent, useEffect, useState } from 'react';
import { AlertTriangle, ShieldCheck, Activity, ArrowRight, ArrowLeft } from 'lucide-react';

export default function Sidebar({ 
  selectedAccount, 
  onClose, 
  allLinks = [],
  onResizeStart
}: { 
  selectedAccount: any, 
  onClose: () => void,
  allLinks?: any[],
  onResizeStart: (event: PointerEvent<HTMLDivElement>) => void
}) {
  const [counterpartyFilter, setCounterpartyFilter] = useState('');
  const isHighRisk = selectedAccount?.risk === 'laundering';
  const isMediumRisk = selectedAccount?.risk === 'suspicious';

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
    <div className="relative h-screen w-full overflow-hidden border-l border-slate-200 bg-white p-6 text-slate-800 shadow-2xl transition-colors duration-300 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200">
      <div
        role="separator"
        aria-label="Resize sidebar"
        aria-orientation="vertical"
        onPointerDown={onResizeStart}
        className="absolute left-0 top-0 h-full w-3 -translate-x-1/2 cursor-col-resize touch-none"
      >
        <div className="absolute inset-y-0 left-1/2 w-px -translate-x-1/2 bg-slate-200 transition-colors hover:bg-slate-400 dark:bg-slate-700 dark:hover:bg-slate-500" />
      </div>

      <div className="flex justify-between items-center shrink-0">
        <h2 className="text-xl font-bold tracking-tight text-slate-900 dark:text-white">Account Details</h2>
        <button onClick={onClose} className="text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-white transition-colors">&times;</button>
      </div>

      {selectedAccount && (
        <div className="flex flex-col h-full overflow-hidden space-y-6">
          
          <div className="space-y-6 shrink-0">
            <div>
              <p className="text-sm text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-1">Account ID</p>
              <p className="text-2xl font-mono text-slate-900 dark:text-white">{selectedAccount.id}</p>
              
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
              <h3 className="text-sm text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2">Pattern Detected</h3>
              <p className="font-semibold text-slate-800 dark:text-slate-100">{selectedAccount.pattern}</p>
            </div>

            <div>
              <h3 className="text-sm text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2">AI Explanation</h3>
              <p className="text-sm leading-relaxed text-slate-700 dark:text-slate-300">
                {selectedAccount.aiExplanation}
              </p>
            </div>
          </div>

          {/* Transactions List remains exactly the same */}
          <div className="flex flex-col flex-1 min-h-0 border-t border-slate-200 dark:border-slate-700 pt-6 transition-colors">
            <div className="mb-4 shrink-0">
              <label
                htmlFor="counterparty-filter"
                className="mb-2 block text-sm text-slate-500 dark:text-slate-400 uppercase tracking-wider"
              >
                Filter User
              </label>
              <input
                id="counterparty-filter"
                type="text"
                value={counterpartyFilter}
                onChange={(e) => setCounterpartyFilter(e.target.value)}
                placeholder="Type an account ID"
                className="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-900 outline-none transition-colors placeholder-slate-400 focus:border-slate-400 dark:border-slate-700 dark:bg-slate-800 dark:text-white dark:placeholder-slate-500 dark:focus:border-slate-500"
              />
            </div>

            <div className="flex justify-between items-end mb-3 shrink-0">
              <h3 className="text-sm text-slate-500 dark:text-slate-400 uppercase tracking-wider">Known Transactions</h3>
              <span className="text-xs text-slate-500">{filteredTransactions.length} records</span>
            </div>
             
            <div className="flex-1 overflow-y-auto pr-2 space-y-2 pb-4 scrollbar-thin scrollbar-thumb-slate-300 dark:scrollbar-thumb-slate-700 scrollbar-track-transparent">
              {filteredTransactions.length === 0 ? (
                <p className="text-sm text-slate-500 italic">
                  {accountTransactions.length === 0 ? 'No connections found.' : 'No transactions match the selected user.'}
                </p>
              ) : (
                filteredTransactions.map((tx, idx) => {
                  const sourceId = tx.source.id || tx.source;
                  const targetId = tx.target.id || tx.target;
                  const isOutgoing = sourceId === selectedAccount.id;
                  const counterpartId = isOutgoing ? targetId : sourceId;

                  return (
                    <div key={idx} className="bg-slate-50 dark:bg-slate-800 p-3 rounded-lg border border-slate-200 dark:border-slate-700 flex flex-col gap-1 hover:border-slate-300 dark:hover:border-slate-500 transition-colors">
                      <div className="flex justify-between items-center">
                        <div className={`flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider ${isOutgoing ? 'text-orange-500 dark:text-orange-400' : 'text-emerald-600 dark:text-emerald-400'}`}>
                          {isOutgoing ? <ArrowRight size={14} /> : <ArrowLeft size={14} />}
                          {isOutgoing ? 'Sent to' : 'Received from'}
                        </div>
                        <div className="font-mono text-sm font-semibold text-slate-900 dark:text-slate-200">
                          ${tx.amount.toLocaleString()}
                        </div>
                      </div>
                      <div className="font-mono text-sm text-slate-500 dark:text-slate-400 ml-5">
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
