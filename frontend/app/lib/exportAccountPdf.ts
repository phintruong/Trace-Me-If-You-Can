import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';

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

type PdfTransaction = {
  direction: 'Incoming' | 'Outgoing';
  counterpartId: string;
  amount: number;
};

type PdfStats = {
  totalTransactions: number;
  incomingCount: number;
  outgoingCount: number;
  totalIncomingAmount: number;
  totalOutgoingAmount: number;
  uniqueCounterparties: number;
  largestTransaction: number;
  topCounterparties: Array<{
    id: string;
    count: number;
    totalAmount: number;
  }>;
};

type ExportPayload = {
  account: Account;
  transactions: PdfTransaction[];
  stats: PdfStats;
};

function safeText(value: unknown, fallback = 'Not available') {
  if (value === null || value === undefined) return fallback;
  const text = String(value).trim();
  if (!text || text.toUpperCase() === 'TOBEFILLED') return fallback;
  return text;
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

function wrapText(doc: jsPDF, text: string, x: number, y: number, maxWidth: number, lineHeight = 6) {
  const lines = doc.splitTextToSize(text, maxWidth);
  doc.text(lines, x, y);
  return y + lines.length * lineHeight;
}

function getRiskAccent(risk?: string): [number, number, number] {
  if (risk === 'laundering') return [196, 53, 53];
  if (risk === 'suspicious') return [217, 119, 6];
  return [46, 125, 50];
}

export async function exportAccountSummaryPdf({
  account,
  transactions,
  stats,
}: ExportPayload) {
  const doc = new jsPDF({
    orientation: 'p',
    unit: 'mm',
    format: 'a4',
  });

  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();
  const margin = 14;
  const contentWidth = pageWidth - margin * 2;
  const accent = getRiskAccent(account.risk);

  const generatedAt = new Date().toLocaleString();

  doc.setFillColor(18, 24, 38);
  doc.rect(0, 0, pageWidth, 34, 'F');

  doc.setDrawColor(accent[0], accent[1], accent[2]);
  doc.setLineWidth(1.2);
  doc.line(margin, 28, pageWidth - margin, 28);

  doc.setTextColor(255, 248, 220);
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(20);
  doc.text('Account Summary Report', margin, 14);

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(10);
  doc.setTextColor(225, 225, 225);
  doc.text(`Generated: ${generatedAt}`, margin, 21);

  doc.setTextColor(30, 30, 30);

  let y = 42;

  doc.setFillColor(248, 248, 248);
  doc.roundedRect(margin, y, contentWidth, 24, 3, 3, 'F');

  doc.setFont('helvetica', 'bold');
  doc.setFontSize(11);
  doc.text('Account ID', margin + 4, y + 7);

  doc.setFont('courier', 'bold');
  doc.setFontSize(14);
  doc.text(String(account.id), margin + 4, y + 15);

  doc.setFont('helvetica', 'bold');
  doc.setFontSize(11);
  doc.text('Risk Label', pageWidth - 60, y + 7);

  doc.setTextColor(accent[0], accent[1], accent[2]);
  doc.setFontSize(13);
  doc.text(formatRiskLabel(account.risk), pageWidth - 60, y + 15);

  doc.setTextColor(30, 30, 30);
  y += 31;

  autoTable(doc, {
    startY: y,
    theme: 'grid',
    styles: {
      font: 'helvetica',
      fontSize: 9,
      cellPadding: 3.2,
      textColor: [35, 35, 35],
      lineColor: [225, 225, 225],
      lineWidth: 0.25,
    },
    headStyles: {
      fillColor: [30, 41, 59],
      textColor: [255, 255, 255],
      fontStyle: 'bold',
    },
    bodyStyles: {
      fillColor: [252, 252, 252],
    },
    columnStyles: {
      0: { cellWidth: 40 },
      1: { cellWidth: 48 },
      2: { cellWidth: 40 },
      3: { cellWidth: 45 },
    },
    head: [['Risk Score', 'Total Transactions', 'Counterparties', 'Largest Transaction']],
    body: [[
      typeof account.riskScore === 'number' ? account.riskScore.toFixed(4) : 'Not available',
      String(account.txCount ?? stats.totalTransactions),
      String(stats.uniqueCounterparties),
      formatMoney(stats.largestTransaction),
    ]],
  });

  y = (doc as jsPDF & { lastAutoTable?: { finalY: number } }).lastAutoTable?.finalY ?? y;
  y += 8;

  autoTable(doc, {
    startY: y,
    theme: 'grid',
    styles: {
      font: 'helvetica',
      fontSize: 9,
      cellPadding: 3.2,
      textColor: [35, 35, 35],
      lineColor: [225, 225, 225],
      lineWidth: 0.25,
    },
    headStyles: {
      fillColor: [71, 85, 105],
      textColor: [255, 255, 255],
      fontStyle: 'bold',
    },
    head: [['Incoming Count', 'Outgoing Count', 'Incoming Volume', 'Outgoing Volume']],
    body: [[
      String(stats.incomingCount),
      String(stats.outgoingCount),
      formatMoney(stats.totalIncomingAmount),
      formatMoney(stats.totalOutgoingAmount),
    ]],
  });

  y = (doc as jsPDF & { lastAutoTable?: { finalY: number } }).lastAutoTable?.finalY ?? y;
  y += 10;

  doc.setFont('helvetica', 'bold');
  doc.setFontSize(12);
  doc.text('Detected Pattern', margin, y);
  y += 6;

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(10);
  y = wrapText(doc, safeText(account.pattern), margin, y, contentWidth, 5.5);
  y += 4;

  doc.setFont('helvetica', 'bold');
  doc.setFontSize(12);
  doc.text('AI Explanation', margin, y);
  y += 6;

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(10);
  y = wrapText(doc, safeText(account.aiExplanation), margin, y, contentWidth, 5.5);
  y += 4;

  const topCounterpartyRows =
    stats.topCounterparties.length > 0
      ? stats.topCounterparties.map((item) => [
          item.id,
          String(item.count),
          formatMoney(item.totalAmount),
        ])
      : [['No counterparties found', '-', '-']];

  autoTable(doc, {
    startY: y,
    theme: 'striped',
    styles: {
      font: 'helvetica',
      fontSize: 9,
      cellPadding: 3,
      textColor: [35, 35, 35],
    },
    headStyles: {
      fillColor: accent,
      textColor: [255, 255, 255],
      fontStyle: 'bold',
    },
    head: [['Top Counterparty', 'Transactions', 'Total Volume']],
    body: topCounterpartyRows,
  });

  y = (doc as jsPDF & { lastAutoTable?: { finalY: number } }).lastAutoTable?.finalY ?? y;
  y += 8;

  const transactionRows =
    transactions.length > 0
      ? transactions.slice(0, 12).map((tx) => [
          tx.direction,
          tx.counterpartId,
          formatMoney(tx.amount),
        ])
      : [['No transactions found', '-', '-']];

  autoTable(doc, {
    startY: y,
    theme: 'grid',
    styles: {
      font: 'helvetica',
      fontSize: 8.7,
      cellPadding: 2.8,
      textColor: [35, 35, 35],
      lineColor: [228, 228, 228],
      lineWidth: 0.2,
    },
    headStyles: {
      fillColor: [15, 23, 42],
      textColor: [255, 255, 255],
      fontStyle: 'bold',
    },
    head: [['Direction', 'Counterparty', 'Amount']],
    body: transactionRows,
  });

  const totalPages = doc.getNumberOfPages();
  for (let i = 1; i <= totalPages; i += 1) {
    doc.setPage(i);
    doc.setDrawColor(230, 230, 230);
    doc.line(margin, pageHeight - 10, pageWidth - margin, pageHeight - 10);
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(9);
    doc.setTextColor(90, 90, 90);
    doc.text(`AML Network Report • Page ${i} of ${totalPages}`, margin, pageHeight - 5);
  }

  const fileName = `account-summary-${account.id}.pdf`;
  doc.save(fileName);
}
