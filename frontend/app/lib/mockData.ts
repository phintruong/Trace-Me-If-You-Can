export const graphData = {
  nodes: [
    { id: 'ACC-1001', risk: 'normal', txCount: 12, pattern: 'None', aiExplanation: 'Standard salary deposits and utility payments.' },
    { id: 'ACC-1002', risk: 'suspicious', txCount: 45, pattern: 'Structuring', aiExplanation: 'Multiple deposits just below the $10,000 reporting threshold over a 5-day period.' },
    { id: 'ACC-1003', risk: 'laundering', txCount: 130, pattern: 'Smurfing / Rapid Movement', aiExplanation: 'Acts as a central node. Receives funds from dozens of accounts and immediately wires them to offshore entities.' },
    { id: 'ACC-1004', risk: 'normal', txCount: 8, pattern: 'None', aiExplanation: 'Standard retail spending behavior.' },
    { id: 'ACC-1005', risk: 'suspicious', txCount: 22, pattern: 'Pass-through', aiExplanation: 'Funds are held for less than 24 hours before being forwarded to ACC-1003.' },
  ],
  links: [
    { source: 'ACC-1001', target: 'ACC-1002', amount: 500 },
    { source: 'ACC-1002', target: 'ACC-1003', amount: 9500 },
    { source: 'ACC-1004', target: 'ACC-1005', amount: 1200 },
    { source: 'ACC-1005', target: 'ACC-1003', amount: 8800 },
    { source: 'ACC-1003', target: 'ACC-1001', amount: 200 },
  ]
};