"""Export API: PDF and LaTeX reports for flagged accounts."""

import io
import logging
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.services import export_service

router = APIRouter(prefix="/export", tags=["export"])
logger = logging.getLogger(__name__)


@router.get("/flagged-accounts/pdf")
def export_flagged_accounts_pdf(
    risk_threshold: float = Query(0.5, ge=0.0, le=1.0, description="Minimum risk score to include"),
    limit: Optional[int] = Query(None, ge=1, description="Max number of accounts to include"),
    include_ai_explanations: bool = Query(True, description="Include Watsonx AI explanations"),
    format: Literal["pdf", "latex"] = Query("pdf", description="Report format: pdf or latex"),
):
    """
    Generate and download a PDF or LaTeX report of flagged accounts.
    Returns file as attachment with timestamped filename.
    """
    try:
        content, filename = export_service.generate_flagged_accounts_report(
            risk_threshold=risk_threshold,
            limit=limit,
            include_ai_explanations=include_ai_explanations,
            format=format,
        )
    except ValueError as e:
        if "No flagged accounts" in str(e):
            raise HTTPException(status_code=404, detail=str(e)) from e
        raise HTTPException(status_code=422, detail=str(e)) from e
    except RuntimeError as e:
        logger.exception("Export failed: %s", e)
        raise HTTPException(status_code=500, detail="Report generation failed.") from e
    except Exception as e:
        logger.exception("Unexpected error during export: %s", e)
        raise HTTPException(status_code=500, detail="Report generation failed.") from e

    if format == "pdf":
        media_type = "application/pdf"
    else:
        media_type = "text/plain"

    return StreamingResponse(
        io.BytesIO(content),
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )
