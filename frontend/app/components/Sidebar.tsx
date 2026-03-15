import { type PointerEvent, useEffect, useMemo, useState } from 'react';
import {
  AlertTriangle,
  ShieldCheck,
  Activity,
  ArrowRight,
  ArrowLeft,
  FileDown,
  Loader2,
  Network,
  DollarSign,
  Hash,
} from 'lucide-react';
import { exportAccountSummaryPdf } from '../lib/exportAccountPdf';

type Account = {
  id: string;
  risk?: string;
  riskScore?: number;
  txCount?: number;
  pattern?: string;
  aiExplanation?: string;
  role?: string;
  cluster?: number | string;
};

type LinkShape = {
  source: string | { id: string };
  target: string | { id: string };
  amount?: number;
};

function getNodeId(node: string | { id: string }) {
  return typeof node === 'string' ? node : node?.id;
}

function formatMoney(value: number) {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 2,
  }).format(value || 0);
}

function formatRiskLabel(risk?: string) {
  if (!risk) return 'UNKNOWN';
  return risk.replace(/_/g, ' ').toUpperCase();
}

function safeText(value: unknown, fallback = 'Not available') {
  if (value === null || value === undefined) return fallback;
  const text = String(value).trim();
  if (!text || text.toUpperCase() === 'TOBEFILLED') return fallback;
  return text;
}

export default function Sidebar({
  selectedAccount,
  onClose,
  allLinks = [],
  onResizeStart,
  isDarkMode,
}: {
  selectedAccount: Account | null;
  onClose: () => void;
  allLinks?: LinkShape[];
  onResizeStart: (event: PointerEvent<HTMLDivElement>) => void;
  isDarkMode: boolean;
}) {
  const [counterpartyFilter, setCounterpartyFilter] = useState('');
  const [isExportingPdf, setIsExportingPdf] = useState(false);

  const isHighRisk = selectedAccount?.risk === 'laundering';
  const isMediumRisk = selectedAccount?.risk === 'suspicious';

  const panelClasses = isDarkMode
    ? 'border-[#5c4a11] bg-[#21181f] text-[#fdf4c6]'
    : 'border-[#e7da7d] bg-[#fffbe0] text-[#2D1E2F]';

  const titleClasses = isDarkMode ? 'text-[#fff7cc]' : 'text-[#2D1E2F]';
  const mutedTextClasses = isDarkMode ? 'text-[#d9c874]' : 'text-[#6a5a35]';
  const bodyTextClasses = isDarkMode ? 'text-[#f0df9b]' : 'text-[#4b394d]';

  const surfaceClasses = isDarkMode
    ? 'border-[#5c4a11] bg-[#2c2129]'
    : 'border-[#efe39a] bg-[#fffdf1]';

  const resizeHandleClasses = isDarkMode
    ? 'bg-[#705f19] hover:bg-[#8b7720]'
    : 'bg-[#efe39a] hover:bg-[#e7da7d]';

  const accountTransactions = useMemo(() => {
    if (!selectedAccount?.id) return [];
    return allLinks.filter((link) => {
      const sourceId = getNodeId(link.source);
      const targetId = getNodeId(link.target);
      return sourceId === selectedAccount.id || targetId === selectedAccount.id;
    });
  }, [allLinks, selectedAccount?.id]);

  const stats = useMemo(() => {
    if (!selectedAccount?.id) {
      return {
        totalTransactions: 0,
        incomingCount: 0,
        outgoingCount: 0,
        totalIncomingAmount: 0,
        totalOutgoingAmount: 0,
        uniqueCounterparties: 0,
        largestTransaction: 0,
        topCounterparties: [] as Array<{
          id: string;
          count: number;
          totalAmount: number;
        }>,
      };
    }

    let incomingCount = 0;
    let outgoingCount = 0;
    let totalIncomingAmount = 0;
    let totalOutgoingAmount = 0;
    let largestTransaction = 0;

    const counterpartyMap = new Map<
      string,
      { count: number; totalAmount: number }
    >();

    for (const tx of accountTransactions) {
      const sourceId = getNodeId(tx.source);
      const targetId = getNodeId(tx.target);
      const amount = Number(tx.amount || 0);
      const isOutgoing = sourceId === selectedAccount.id;
      const counterpartId = isOutgoing ? targetId : sourceId;

      if (!counterpartId) continue;

      if (isOutgoing) {
        outgoingCount += 1;
        totalOutgoingAmount += amount;
      } else {
        incomingCount += 1;
        totalIncomingAmount += amount;
      }

      largestTransaction = Math.max(largestTransaction, amount);

      const prev = counterpartyMap.get(counterpartId) ?? {
        count: 0,
        totalAmount: 0,
      };
      counterpartyMap.set(counterpartId, {
        count: prev.count + 1,
        totalAmount: prev.totalAmount + amount,
      });
    }

    const topCounterparties = [...counterpartyMap.entries()]
      .map(([id, value]) => ({
        id,
        count: value.count,
        totalAmount: value.totalAmount,
      }))
      .sort((a, b) => {
        if (b.totalAmount !== a.totalAmount) return b.totalAmount - a.totalAmount;
        return b.count - a.count;
      })
      .slice(0, 5);

    return {
      totalTransactions: accountTransactions.length,
      incomingCount,
      outgoingCount,
      totalIncomingAmount,
      totalOutgoingAmount,
      uniqueCounterparties: counterpartyMap.size,
      largestTransaction,
      topCounterparties,
    };
  }, [accountTransactions, selectedAccount?.id]);

  const filteredTransactions = useMemo(() => {
    if (!counterpartyFilter.trim()) return accountTransactions;

    return accountTransactions.filter((link) => {
      const sourceId = getNodeId(link.source);
      const targetId = getNodeId(link.target);
      const counterpartId =
        sourceId === selectedAccount?.id ? targetId : sourceId;

      return String(counterpartId || '')
        .toLowerCase()
        .includes(counterpartyFilter.trim().toLowerCase());
    });
  }, [accountTransactions, counterpartyFilter, selectedAccount?.id]);

  useEffect(() => {
    setCounterpartyFilter('');
  }, [selectedAccount?.id]);

  const handleExportPdf = async () => {
    if (!selectedAccount) return;

    try {
      setIsExportingPdf(true);
      await exportAccountSummaryPdf({
        account: selectedAccount,
        transactions: accountTransactions.map((tx) => {
          const sourceId = getNodeId(tx.source);
          const targetId = getNodeId(tx.target);
          const isOutgoing = sourceId === selectedAccount.id;
          const counterpartId = isOutgoing ? targetId : sourceId;

          return {
            direction: isOutgoing ? 'Outgoing' : 'Incoming',
            counterpartId: String(counterpartId || ''),
            amount: Number(tx.amount || 0),
          };
        }),
        stats,
      });
    } catch (error) {
      console.error('Failed to export PDF', error);
      alert('Could not generate the PDF.');
    } finally {
      setIsExportingPdf(false);
    }
  };

  return (
    <div
      className={`relative h-full w-full overflow-hidden border-l p-6 shadow-2xl transition-colors duration-300 ${panelClasses}`}
    >
      <div
        role="separator"
        aria-label="Resize sidebar"
        aria-orientation="vertical"
        onPointerDown={onResizeStart}
        className="absolute left-0 top-0 h-full w-3 -translate-x-1/2 cursor-col-resize touch-none"
      >
        <div
          className={`absolute inset-y-0 left-1/2 w-px -translate-x-1/2 transition-colors ${resizeHandleClasses}`}
        />
      </div>

      <div className="flex h-full flex-col overflow-hidden">
        <div className="mb-5 flex shrink-0 items-start justify-between gap-3">
          <div>
            <h2 className={`text-xl font-bold tracking-tight ${titleClasses}`}>
              Account Details
            </h2>
            <p className={`mt-1 text-sm ${mutedTextClasses}`}>
              Review account signals and export a polished PDF summary.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleExportPdf}
              disabled={!selectedAccount || isExportingPdf}
              className={`group relative inline-flex items-center gap-2 rounded-xl border px-4 py-2.5 text-sm font-semibold tracking-wide transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60 disabled:shadow-none ${
                isDarkMode
                  ? 'border-[#8b7720] bg-linear-to-br from-[#3a2d20] to-[#2b2128] text-[#fff7cc] shadow-[0_8px_20px_rgba(139,119,32,0.25)] hover:-translate-y-0.5 hover:from-[#463524] hover:to-[#332734] hover:shadow-[0_10px_24px_rgba(139,119,32,0.35)] focus-visible:ring-[#d9c874] focus-visible:ring-offset-[#21181f]'
                  : 'border-[#d7c454] bg-linear-to-br from-[#fff9d4] to-[#ffeeb5] text-[#2D1E2F] shadow-[0_8px_18px_rgba(199,173,60,0.28)] hover:-translate-y-0.5 hover:from-[#fff3bf] hover:to-[#ffe39a] hover:shadow-[0_10px_24px_rgba(199,173,60,0.35)] focus-visible:ring-[#b8a63f] focus-visible:ring-offset-[#fffbe0]'
              }`}
              aria-label="Export account summary as PDF"
            >
              {isExportingPdf ? (
                <Loader2 size={16} className="animate-spin" />
              ) : (
                <FileDown
                  size={16}
                  className=""
                />
              )}
              {/* {isExportingPdf ? 'Preparing PDF...' : 'Export PDF'} */}
            </button>

            <button
              onClick={onClose}
              className={`rounded-md p-2 text-4xl leading-none transition-colors ${
                isDarkMode
                  ? 'text-[#d9c874] hover:bg-[#2c2129] hover:text-[#fff7cc]'
                  : 'text-[#6a5a35] hover:bg-[#fff7cc] hover:text-[#2D1E2F]'
              }`}
              aria-label="Close sidebar"
            >
              &times;
            </button>
          </div>
        </div>

        {selectedAccount && (
          <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
            <div className="shrink-0 space-y-5">
              <div
                className={`rounded-2xl border p-4 shadow-sm transition-colors ${surfaceClasses}`}
              >
                <p
                  className={`mb-1 text-xs uppercase tracking-[0.18em] ${mutedTextClasses}`}
                >
                  Account ID
                </p>
                <p className={`break-all font-mono text-xl ${titleClasses}`}>
                  {selectedAccount.id}
                </p>

                <div
                  className={`mt-3 inline-flex items-center gap-2 rounded-full px-3 py-1 text-sm font-semibold
                  ${
                    isHighRisk
                      ? 'border border-[#e3170a]/30 bg-[#e3170a]/12 text-[#e3170a] dark:border-[#8f0e08]/60 dark:bg-[#8f0e08]/22 dark:text-[#e06b62]'
                      : isMediumRisk
                        ? 'border border-[#F78D2A]/30 bg-[#F78D2A]/14 text-[#b85f14] dark:border-[#B85F14]/60 dark:bg-[#B85F14]/22 dark:text-[#f0a55f]'
                        : 'border border-[#a9e5bb]/45 bg-[#a9e5bb]/30 text-[#4d6f56] dark:border-[#5B8A68]/60 dark:bg-[#5B8A68]/24 dark:text-[#b5d8bf]'
                  }`}
                >
                  {isHighRisk && <AlertTriangle size={16} />}
                  {isMediumRisk && <Activity size={16} />}
                  {!isHighRisk && !isMediumRisk && <ShieldCheck size={16} />}
                  {formatRiskLabel(selectedAccount.risk)}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div
                  className={`rounded-2xl border p-4 transition-colors ${surfaceClasses}`}
                >
                  <div className={`mb-2 flex items-center gap-2 ${mutedTextClasses}`}>
                    <DollarSign size={16} />
                    <span className="text-xs uppercase tracking-[0.16em]">
                      Largest Tx
                    </span>
                  </div>
                  <p className={`text-lg font-semibold ${titleClasses}`}>
                    {formatMoney(stats.largestTransaction)}
                  </p>
                </div>

                <div
                  className={`rounded-2xl border p-4 transition-colors ${surfaceClasses}`}
                >
                  <div className={`mb-2 flex items-center gap-2 ${mutedTextClasses}`}>
                    <Network size={16} />
                    <span className="text-xs uppercase tracking-[0.16em]">
                      Counterparties
                    </span>
                  </div>
                  <p className={`text-lg font-semibold ${titleClasses}`}>
                    {stats.uniqueCounterparties}
                  </p>
                </div>

                <div
                  className={`rounded-2xl border p-4 transition-colors ${surfaceClasses}`}
                >
                  <div className={`mb-2 flex items-center gap-2 ${mutedTextClasses}`}>
                    <ArrowRight size={16} />
                    <span className="text-xs uppercase tracking-[0.16em]">
                      Outgoing
                    </span>
                  </div>
                  <p className={`text-lg font-semibold ${titleClasses}`}>
                    {stats.outgoingCount}
                  </p>
                  <p className={`mt-1 text-xs ${mutedTextClasses}`}>
                    {formatMoney(stats.totalOutgoingAmount)}
                  </p>
                </div>

                <div
                  className={`rounded-2xl border p-4 transition-colors ${surfaceClasses}`}
                >
                  <div className={`mb-2 flex items-center gap-2 ${mutedTextClasses}`}>
                    <ArrowLeft size={16} />
                    <span className="text-xs uppercase tracking-[0.16em]">
                      Incoming
                    </span>
                  </div>
                  <p className={`text-lg font-semibold ${titleClasses}`}>
                    {stats.incomingCount}
                  </p>
                  <p className={`mt-1 text-xs ${mutedTextClasses}`}>
                    {formatMoney(stats.totalIncomingAmount)}
                  </p>
                </div>
              </div>

              <div
                className={`rounded-2xl border p-4 transition-colors ${surfaceClasses}`}
              >
                <h3
                  className={`mb-3 text-sm uppercase tracking-[0.16em] ${mutedTextClasses}`}
                >
                  Summary
                </h3>

                <div className="grid grid-cols-2 gap-x-4 gap-y-3 text-sm">
                  <div>
                    <p className={mutedTextClasses}>Risk score</p>
                    <p className={`font-semibold ${titleClasses}`}>
                      {typeof selectedAccount.riskScore === 'number'
                        ? selectedAccount.riskScore.toFixed(4)
                        : 'Not available'}
                    </p>
                  </div>

                  <div>
                    <p className={mutedTextClasses}>Transaction count</p>
                    <p className={`font-semibold ${titleClasses}`}>
                      {selectedAccount.txCount ?? stats.totalTransactions}
                    </p>
                  </div>

                  <div>
                    <p className={mutedTextClasses}>Role</p>
                    <p className={`font-semibold ${titleClasses}`}>
                      {safeText(selectedAccount.role)}
                    </p>
                  </div>

                  <div>
                    <p className={mutedTextClasses}>Cluster</p>
                    <p className={`font-semibold ${titleClasses}`}>
                      {selectedAccount.cluster ?? 'Not available'}
                    </p>
                  </div>
                </div>
              </div>

              <div
                className={`rounded-2xl border p-4 transition-colors ${surfaceClasses}`}
              >
                <h3
                  className={`mb-2 text-sm uppercase tracking-[0.16em] ${mutedTextClasses}`}
                >
                  Pattern Detected
                </h3>
                <p className={`font-semibold ${titleClasses}`}>
                  {safeText(selectedAccount.pattern)}
                </p>
              </div>

              <div
                className={`rounded-2xl border p-4 transition-colors ${surfaceClasses}`}
              >
                <h3
                  className={`mb-2 text-sm uppercase tracking-[0.16em] ${mutedTextClasses}`}
                >
                  AI Explanation
                </h3>
                <p className={`text-sm leading-relaxed ${bodyTextClasses}`}>
                  {safeText(selectedAccount.aiExplanation)}
                </p>
              </div>

              <div
                className={`rounded-2xl border p-4 transition-colors ${surfaceClasses}`}
              >
                <div className="mb-3 flex items-center justify-between">
                  <h3
                    className={`text-sm uppercase tracking-[0.16em] ${mutedTextClasses}`}
                  >
                    Top Counterparties
                  </h3>
                  <span className={`text-xs ${mutedTextClasses}`}>
                    Top 5 by volume
                  </span>
                </div>

                {stats.topCounterparties.length === 0 ? (
                  <p className={`text-sm italic ${mutedTextClasses}`}>
                    No counterparties found.
                  </p>
                ) : (
                  <div className="space-y-2">
                    {stats.topCounterparties.map((item) => (
                      <div
                        key={item.id}
                        className={`rounded-xl border px-3 py-2 ${isDarkMode ? 'border-[#4d3d14]' : 'border-[#efe39a]'}`}
                      >
                        <div className="flex items-center justify-between gap-3">
                          <div className="min-w-0">
                            <p className={`truncate font-mono text-sm ${titleClasses}`}>
                              {item.id}
                            </p>
                            <p className={`text-xs ${mutedTextClasses}`}>
                              {item.count} transaction{item.count === 1 ? '' : 's'}
                            </p>
                          </div>
                          <p className={`text-sm font-semibold ${titleClasses}`}>
                            {formatMoney(item.totalAmount)}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            <div
              className={`mt-6 flex min-h-0 flex-1 flex-col border-t pt-6 transition-colors ${
                isDarkMode ? 'border-[#5c4a11]' : 'border-[#efe39a]'
              }`}
            >
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
                      ? 'border-[#5c4a11] bg-[#2c2129] text-[#fff7cc] placeholder-[#b7a867] focus:border-[#d9c874]'
                      : 'border-[#e7da7d] bg-[#fffdf1] text-[#2D1E2F] placeholder-[#978850] focus:border-[#cdbf5e]'
                  }`}
                />
              </div>

              <div className="mb-3 flex shrink-0 items-end justify-between">
                <h3 className={`text-sm uppercase tracking-wider ${mutedTextClasses}`}>
                  Known Transactions
                </h3>
                <span className={`text-xs ${mutedTextClasses}`}>
                  {filteredTransactions.length} records
                </span>
              </div>

              <div className="scrollbar-thin scrollbar-track-transparent scrollbar-thumb-[#efe39a] dark:scrollbar-thumb-[#705f19] flex-1 space-y-2 overflow-y-auto pb-4 pr-2">
                {filteredTransactions.length === 0 ? (
                  <p className={`text-sm italic ${mutedTextClasses}`}>
                    {accountTransactions.length === 0
                      ? 'No connections found.'
                      : 'No transactions match the selected user.'}
                  </p>
                ) : (
                  filteredTransactions.map((tx, idx) => {
                    const sourceId = getNodeId(tx.source);
                    const targetId = getNodeId(tx.target);
                    const isOutgoing = sourceId === selectedAccount.id;
                    const counterpartId = isOutgoing ? targetId : sourceId;

                    return (
                      <div
                        key={`${sourceId}-${targetId}-${idx}`}
                        className={`flex flex-col gap-1 rounded-lg border p-3 transition-colors ${
                          isDarkMode
                            ? `${surfaceClasses} hover:border-[#8b7720]`
                            : `${surfaceClasses} hover:border-[#e7da7d]`
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <div
                            className={`flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider ${
                              isOutgoing
                                ? 'text-[#b85f14] dark:text-[#f0a55f]'
                                : 'text-[#5c7c66] dark:text-[#b5d8bf]'
                            }`}
                          >
                            {isOutgoing ? <ArrowRight size={14} /> : <ArrowLeft size={14} />}
                            {isOutgoing ? 'Sent to' : 'Received from'}
                          </div>
                          <div
                            className={`font-mono text-sm font-semibold ${
                              isDarkMode ? 'text-slate-200' : 'text-stone-950'
                            }`}
                          >
                            {formatMoney(Number(tx.amount || 0))}
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

        {!selectedAccount && (
          <div className="flex h-full items-center justify-center">
            <div className={`text-center ${mutedTextClasses}`}>
              <Hash className="mx-auto mb-3" size={22} />
              <p>Select an account to inspect its details.</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
