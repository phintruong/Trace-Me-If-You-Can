"""PDF and LaTeX export for flagged accounts reports."""

import io
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)

from app.services import db_client

logger = logging.getLogger(__name__)

# Risk level thresholds for color coding (aligned with account.py)
RISK_GREEN = 0.5   # green < 0.5
RISK_YELLOW = 0.7  # yellow 0.5-0.7
RISK_ORANGE = 0.9  # orange 0.7-0.9
LAUNDERING_THRESHOLD = 0.9  # red >= 0.9


def _risk_color(score: float) -> Tuple[float, float, float]:
    """Return (r, g, b) for risk score: green, yellow, orange, red."""
    if score < RISK_GREEN:
        return (0.2, 0.7, 0.3)   # green
    if score < RISK_YELLOW:
        return (0.9, 0.85, 0.2)  # yellow
    if score < RISK_ORANGE:
        return (1.0, 0.55, 0.2)  # orange
    return (0.9, 0.2, 0.2)  # red


def _format_ts() -> str:
    """Return timestamp string for filename and report: YYYYMMDD_HHMMSS."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _build_pdf(
    accounts: List[Dict[str, Any]],
    include_ai_explanations: bool,
    generated_ts: str,
) -> bytes:
    """Build PDF buffer and return bytes."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=inch,
        leftMargin=inch,
        topMargin=inch,
        bottomMargin=0.75 * inch,
    )
    styles = getSampleStyleSheet()
    story = []

    # Title page
    story.append(Spacer(1, 1.5 * inch))
    story.append(Paragraph(
        "<b>Flagged Accounts Report</b>",
        styles["Title"],
    ))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(
        f"Generated: {generated_ts.replace('_', ' ')} UTC",
        styles["Normal"],
    ))
    story.append(Paragraph(
        "Fraud detection system – for analyst use",
        styles["Normal"],
    ))
    story.append(Spacer(1, 2 * inch))
    story.append(PageBreak())

    # Executive summary
    total = len(accounts)
    avg_risk = sum(a["risk_score"] for a in accounts) / total if total else 0
    high_risk_count = sum(1 for a in accounts if a["risk_score"] >= LAUNDERING_THRESHOLD)
    summary_data = [
        ["Metric", "Value"],
        ["Total flagged accounts", str(total)],
        ["Average risk score", f"{avg_risk:.2f}"],
        ["High-risk accounts (score ≥ 0.9)", str(high_risk_count)],
    ]
    t_summary = Table(summary_data, colWidths=[3 * inch, 2.5 * inch])
    t_summary.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F2F2")]),
    ]))
    story.append(Paragraph("Executive Summary", styles["Heading1"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(t_summary)
    story.append(Spacer(1, 0.4 * inch))

    # Detailed account list
    story.append(Paragraph("Flagged Account Details", styles["Heading1"]))
    story.append(Spacer(1, 0.2 * inch))

    for i, acc in enumerate(accounts):
        risk = acc["risk_score"]
        r, g, b = _risk_color(risk)
        risk_color = colors.Color(r, g, b)
        account_id = acc.get("account_id", "")
        summary_text = acc.get("summary_text")
        if not summary_text or not str(summary_text).strip():
            summary_text = "No explanation available."
        else:
            summary_text = str(summary_text).strip()

        # Account block
        detail_data = [
            ["Account ID", str(account_id)],
            ["Risk score", f"{risk:.2f}"],
        ]
        if acc.get("transaction_count") is not None:
            detail_data.append(["Transaction count", str(acc["transaction_count"])])
        if acc.get("total_amount") is not None:
            detail_data.append(["Total amount", f"{acc['total_amount']:.2f}"])
        if acc.get("last_transaction_date"):
            detail_data.append(["Last transaction", str(acc["last_transaction_date"])])

        t_acc = Table(detail_data, colWidths=[1.8 * inch, 4 * inch])
        t_acc.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E8E8E8")),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("BACKGROUND", (1, 0), (1, 0), risk_color),
            ("TEXTCOLOR", (1, 0), (1, 0), colors.white),
        ]))
        story.append(t_acc)
        if include_ai_explanations:
            story.append(Spacer(1, 0.1 * inch))
            story.append(Paragraph("<b>AI explanation</b>", styles["Normal"]))
            # Escape HTML in explanation for Paragraph
            safe_text = summary_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            story.append(Paragraph(safe_text, styles["Normal"]))
        story.append(Spacer(1, 0.35 * inch))

    # Notes / methodology
    story.append(PageBreak())
    story.append(Paragraph("Notes & Methodology", styles["Heading1"]))
    story.append(Spacer(1, 0.2 * inch))
    methodology = """
    <b>Risk scores</b> are produced by the fraud detection model and range from 0 to 1.
    Accounts are flagged when at least one transaction meets or exceeds the selected risk threshold.
    <br/><br/>
    <b>Color coding:</b> Green (&lt; 0.5), Yellow (0.5–0.7), Orange (0.7–0.9), Red (≥ 0.9).
    High-risk (≥ 0.9) may indicate laundering; scores ≥ 0.7 are treated as suspicious.
    <br/><br/>
    <b>AI explanations</b> are generated by Watsonx and cached per transaction; they summarize
    why the highest-risk transaction for each account was flagged.
    <br/><br/>
    <i>This report is for internal fraud analysis only. It does not constitute legal or
    regulatory advice.</i>
    """
    story.append(Paragraph(methodology, styles["Normal"]))

    doc.build(
        story,
        onFirstPage=lambda canvas, _: _add_footer(canvas, doc, generated_ts),
        onLaterPages=lambda canvas, _: _add_footer(canvas, doc, generated_ts),
    )
    return buffer.getvalue()


def _add_footer(canvas, doc, generated_ts: str) -> None:
    """Add header and page number to each page."""
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(inch, 0.5 * inch, "Flagged Accounts Report – Generated %s UTC" % generated_ts.replace("_", " "))
    canvas.drawRightString(letter[0] - inch, 0.5 * inch, "Page %d" % canvas.getPageNumber())
    canvas.restoreState()


def _build_latex(
    accounts: List[Dict[str, Any]],
    include_ai_explanations: bool,
    generated_ts: str,
) -> str:
    """Build LaTeX source string."""
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{xcolor}",
        r"\usepackage{colortbl}",
        r"\usepackage{longtable}",
        r"\geometry{letterpaper, margin=1in}",
        r"\title{Flagged Accounts Report}",
        r"\date{Generated: " + generated_ts.replace("_", " ") + r" UTC}",
        r"\begin{document}",
        r"\maketitle",
        r"\section{Executive Summary}",
        r"\begin{tabular}{ll}",
        r"\toprule",
        r"Metric & Value \\",
        r"\midrule",
        "Total flagged accounts & %d \\\\" % len(accounts),
        "Average risk score & %.2f \\\\" % (sum(a["risk_score"] for a in accounts) / len(accounts) if accounts else 0),
        "High-risk accounts (score $\\geq$ 0.9) & %d \\\\" % sum(1 for a in accounts if a["risk_score"] >= LAUNDERING_THRESHOLD),
        r"\bottomrule",
        r"\end{tabular}",
        r"\section{Flagged Account Details}",
    ]
    for acc in accounts:
        risk = acc["risk_score"]
        if risk < RISK_GREEN:
            cellcolor = "green!25"
        elif risk < RISK_YELLOW:
            cellcolor = "yellow!25"
        elif risk < RISK_ORANGE:
            cellcolor = "orange!25"
        else:
            cellcolor = "red!25"
        lines.append(r"\subsection*{Account %s}" % _tex_escape(str(acc.get("account_id", ""))))
        lines.append(r"\begin{tabular}{ll}")
        lines.append("Risk score & \\cellcolor{%s} %.2f \\\\" % (cellcolor, risk))
        if acc.get("transaction_count") is not None:
            lines.append("Transaction count & %d \\\\" % acc["transaction_count"])
        if acc.get("total_amount") is not None:
            lines.append("Total amount & %.2f \\\\" % acc["total_amount"])
        if acc.get("last_transaction_date"):
            lines.append("Last transaction & %s \\\\" % _tex_escape(str(acc["last_transaction_date"])))
        lines.append(r"\end{tabular}")
        if include_ai_explanations:
            summary = acc.get("summary_text") or "No explanation available."
            lines.append(r"\textbf{AI explanation:} " + _tex_escape(str(summary).strip()))
        lines.append("")
    lines.extend([
        r"\section{Notes \& Methodology}",
        r"Risk scores are produced by the fraud detection model (0--1). "
        r"Accounts are flagged when at least one transaction meets the risk threshold. "
        r"Color coding: Green ($<0.5$), Yellow (0.5--0.7), Orange (0.7--0.9), Red ($\geq 0.9$). "
        r"AI explanations are from Watsonx. \textit{For internal use only; not legal advice.}",
        r"\end{document}",
    ])
    return "\n".join(lines)


def _tex_escape(s: str) -> str:
    """Escape special characters for LaTeX."""
    return (
        s.replace("\\", "\\textbackslash ")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("~", "\\textasciitilde ")
        .replace("^", "\\textasciicircum ")
    )


def generate_flagged_accounts_report(
    risk_threshold: float,
    limit: Optional[int],
    include_ai_explanations: bool,
    format: Literal["pdf", "latex"],
) -> Tuple[bytes, str]:
    """
    Generate a flagged accounts report as PDF or LaTeX.
    Returns (file_content_bytes, suggested_filename).
    Raises ValueError if no flagged accounts found.
    """
    generated_ts = _format_ts()
    accounts = db_client.get_flagged_accounts(threshold=risk_threshold, limit=limit)
    if not accounts:
        raise ValueError("No flagged accounts found for the given criteria.")

    # Sort by risk descending (DB already does this; ensure consistent)
    accounts = sorted(accounts, key=lambda a: a["risk_score"], reverse=True)

    if format == "pdf":
        content = _build_pdf(accounts, include_ai_explanations, generated_ts)
        filename = "flagged_accounts_report_%s.pdf" % generated_ts
    else:
        tex = _build_latex(accounts, include_ai_explanations, generated_ts)
        content = tex.encode("utf-8")
        filename = "flagged_accounts_report_%s.tex" % generated_ts

    logger.info("Generated %s report with %d accounts", format, len(accounts))
    return (content, filename)
