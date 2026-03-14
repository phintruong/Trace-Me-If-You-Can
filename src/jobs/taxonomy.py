"""
Normalized job taxonomy: maps raw occupation codes to 5 risk categories.

Categories:
    CASH_HEAVY — occupations where cash handling is routine
    SALARIED   — regular-payroll office/professional workers
    BUSINESS   — owners, traders, self-employed, brokers
    STUDENT    — full-time students
    UNKNOWN    — retired, unemployed, other, unmatched codes
"""

import logging
import pandas as pd

from src.config import DATA_RAW

log = logging.getLogger("aml_pipeline")

# ---------------------------------------------------------------------------
# Keyword-based classification of occupation titles
# ---------------------------------------------------------------------------
# Each list contains substrings matched case-insensitively against occupation_title.

_CASH_HEAVY_KEYWORDS = [
    "food and beverage server", "food counter attendant", "kitchen helper",
    "cashier", "sales support", "retail salesperson",
    "taxi", "limousine", "chauffeur",
    "cleaner", "barber", "hairstylist",
    "cook", "baker", "chef",
    "courier", "messenger",
    "construction trades helper", "labourer",
    "home child care", "home support worker", "caregiver",
    "security guard",
    "store shelf stocker",
    "mine labourer",
    "painters and decorators",
]

_SALARIED_KEYWORDS = [
    "manager", "supervisor",
    "engineer", "technologist", "technician",
    "accountant", "auditor", "financial analyst", "financial advisor",
    "administrative", "officer", "clerk", "receptionist",
    "nurse", "physician", "physiotherapist", "pharmacy",
    "teacher", "professor", "educator",
    "lawyer", "notary",
    "police", "firefighter",
    "programmer", "software developer", "information systems",
    "computer network", "user support",
    "graphic designer", "illustrator",
    "welder", "electrician", "plumber", "pipefitter",
    "carpenter", "automotive service",
    "transport truck driver", "bus driver", "transit operator",
    "mail and message distribution",
    "assembler", "inspector",
    "legislative and senior management",
    "consulting",
    "advertising", "marketing", "public relation",
    "production logistics",
    "dental", "health diagnosing", "health service",
    "social and community service",
    "early childhood educator",
]

_BUSINESS_KEYWORDS = [
    "real estate agent", "insurance agent", "insurance broker",
    "corporate sales manager", "restaurant and food service manager",
    "technical sales specialist",
    "customer and personal services manager",
]

# Special occupation_code strings (not NOC numeric codes)
_SPECIAL_STUDENT = {"STUDENT"}
_SPECIAL_BUSINESS = {"SELF_EMPLOYED"}
_SPECIAL_UNKNOWN = {"OTHER", "UNKNOWN", "RETIRED", "UNEMPLOYED"}


def _match_keywords(title_lower, keywords):
    """Return True if any keyword substring appears in the title."""
    return any(kw in title_lower for kw in keywords)


def build_occupation_lookup():
    """Load kyc_occupation_codes.csv and return {code_str: title_str} dict."""
    path = DATA_RAW / "kyc" / "kyc_occupation_codes.csv"
    df = pd.read_csv(path, dtype=str)
    return dict(zip(df["occupation_code"].str.strip(), df["occupation_title"].str.strip()))


def classify_job(occupation_code, occupation_lookup):
    """Map a single occupation_code string to a job category.

    Args:
        occupation_code: Raw code from KYC (e.g. "65200", "STUDENT", "OTHER").
        occupation_lookup: Dict from build_occupation_lookup().

    Returns:
        One of: "CASH_HEAVY", "SALARIED", "BUSINESS", "STUDENT", "UNKNOWN".
    """
    code = str(occupation_code).strip().upper() if pd.notna(occupation_code) else "UNKNOWN"

    # Special string codes
    if code in _SPECIAL_STUDENT:
        return "STUDENT"
    if code in _SPECIAL_BUSINESS:
        return "BUSINESS"
    if code in _SPECIAL_UNKNOWN or code == "NAN":
        return "UNKNOWN"

    # Look up the NOC title
    # occupation_lookup keys may not be uppercased — try original code
    title = occupation_lookup.get(str(occupation_code).strip(), "")
    if not title:
        title = occupation_lookup.get(code, "")
    if not title:
        return "UNKNOWN"

    title_lower = title.lower()

    # Order matters: check BUSINESS before SALARIED since "manager" appears in both
    if _match_keywords(title_lower, _BUSINESS_KEYWORDS):
        return "BUSINESS"
    if _match_keywords(title_lower, _CASH_HEAVY_KEYWORDS):
        return "CASH_HEAVY"
    if _match_keywords(title_lower, _SALARIED_KEYWORDS):
        return "SALARIED"

    return "UNKNOWN"


def classify_job_series(occupation_codes, account_types, occupation_lookup):
    """Vectorized classification for a full DataFrame.

    Args:
        occupation_codes: pandas Series of raw occupation codes.
        account_types: pandas Series of "individual" / "business" / "unknown".
        occupation_lookup: Dict from build_occupation_lookup().

    Returns:
        pandas Series of job category strings.
    """
    categories = []
    for occ, acct in zip(occupation_codes, account_types):
        if acct == "business":
            categories.append("BUSINESS")
        else:
            categories.append(classify_job(occ, occupation_lookup))

    result = pd.Series(categories, index=occupation_codes.index)
    counts = result.value_counts()
    log.info("  Job category distribution:")
    for cat in ["CASH_HEAVY", "SALARIED", "BUSINESS", "STUDENT", "UNKNOWN"]:
        log.info(f"    {cat:12s}: {counts.get(cat, 0):,}")
    return result
