import os
import re
from typing import Annotated

from fastapi import FastAPI, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from databricks import sql
from dotenv import load_dotenv
from pydantic import BaseModel

# ─── Load Environment Variables ───────────────────────────────────────────────
load_dotenv()

CATALOG = os.getenv("DATABRICKS_CATALOG")
SCHEMA = os.getenv("DATABRICKS_SCHEMA")  # Should be 'functionalities'


# ─── Pydantic Response Models ────────────────────────────────────────────────
# These give you:
#   1. Auto-generated Swagger docs showing exact response shape
#   2. Type validation on outgoing data
#   3. Clear contract for frontend developers

class HealthResponse(BaseModel):
    status: str
    target_schema: str


class FreshnessResponse(BaseModel):
    """Freshness metrics for a single table."""
    last_modified_at: str | None = None
    latency_hours: float | None = None
    current_version: int | None = None
    last_documented_at: str | None = None


class QualityColumn(BaseModel):
    """Quality metrics for a single column in a table."""
    column_name: str
    data_type: str | None = None
    null_percentage: float | None = None
    distinct_count: int | None = None
    total_rows: int | None = None


class QualityNotFound(BaseModel):
    message: str
    data: list


class ProfileColumn(BaseModel):
    """Statistical profile for a single numeric column."""
    column_name: str
    data_type: str | None = None
    mean: float | None = None
    min_val: float | None = None
    p25: float | None = None
    median: float | None = None
    p75: float | None = None
    max_val: float | None = None


class ProfileNotFound(BaseModel):
    message: str
    data: list


# ─── Initialize App ──────────────────────────────────────────────────────────
app = FastAPI(title="Data Discovery API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

# Regex: Only letters, digits, and underscores are allowed in identifiers.
# This blocks any SQL injection attempt (quotes, semicolons, spaces, etc.)
SAFE_IDENTIFIER = re.compile(r"^[a-zA-Z0-9_]+$")


def validate_identifier(value: str, label: str) -> str:
    """
    Validates that a SQL identifier (catalog/schema/table name) contains
    only safe characters.  Raises HTTP 400 if it looks suspicious.

    WHY: Databricks SQL connector doesn't support parameterized queries
    for table/column identifiers, so we MUST validate before interpolation.
    """
    if not SAFE_IDENTIFIER.match(value):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {label}: '{value}'. Only letters, digits, and underscores are allowed.",
        )
    return value


def get_db_connection():
    """Opens a fresh connection to the Databricks SQL warehouse."""
    return sql.connect(
        server_hostname=os.getenv("DATABRICKS_HOST"),
        http_path=os.getenv("DATABRICKS_HTTP_PATH"),
        access_token=os.getenv("DATABRICKS_TOKEN"),
    )


def get_target_table(stats_type: str) -> str:
    """Builds the fully-qualified stats table name, e.g. catalog.schema.complete_freshness_stats"""
    return f"{CATALOG}.{SCHEMA}.complete_{stats_type}_stats"


def parse_table_path(full_table_path: str) -> tuple[str, str, str]:
    """
    Splits 'catalog.schema.table' into three parts AND validates each part
    to prevent SQL injection.
    """
    parts = full_table_path.split(".")
    if len(parts) != 3:
        raise HTTPException(
            status_code=400,
            detail="Path must be in format: catalog.schema.table",
        )
    catalog = validate_identifier(parts[0], "catalog name")
    schema = validate_identifier(parts[1], "schema name")
    table = validate_identifier(parts[2], "table name")
    return catalog, schema, table


# ─── API Endpoints ────────────────────────────────────────────────────────────

@app.get("/")
def health_check() -> HealthResponse:
    """Simple check to verify the API is running and pointing to the right schema."""
    return HealthResponse(status="online", target_schema=f"{CATALOG}.{SCHEMA}")


@app.get("/freshness/{full_table_path}")
def get_freshness(
    full_table_path: Annotated[str, Path(description="Format: catalog.schema.table")],
) -> FreshnessResponse:
    """Fetches latency, version, and last modified timestamps for a table."""
    catalog, schema, table = parse_table_path(full_table_path)
    target_table = get_target_table("freshness")

    query = f"""
        SELECT last_modified_at, latency_hours, current_version, last_documented_at
        FROM {target_table}
        WHERE catalog_name = '{catalog}' AND schema_name = '{schema}' AND table_name = '{table}'
    """

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                row = cursor.fetchone()
                if not row:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Freshness data not found for {full_table_path}",
                    )
                return FreshnessResponse(**row.asDict())
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/quality/{full_table_path}")
def get_quality(
    full_table_path: Annotated[str, Path(description="Format: catalog.schema.table")],
) -> list[QualityColumn]:
    """Fetches null percentages and distinct counts for all columns of a table."""
    catalog, schema, table = parse_table_path(full_table_path)
    target_table = get_target_table("quality")

    query = f"""
        SELECT column_name, data_type, null_percentage, distinct_count, total_rows
        FROM {target_table}
        WHERE catalog_name = '{catalog}' AND schema_name = '{schema}' AND table_name = '{table}'
    """

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
                if not rows:
                    return []
                return [QualityColumn(**row.asDict()) for row in rows]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/profile/{full_table_path}")
def get_profile(
    full_table_path: Annotated[str, Path(description="Format: catalog.schema.table")],
) -> list[ProfileColumn]:
    """Fetches statistical distribution (Mean, Median, p25, p75) for numeric columns."""
    catalog, schema, table = parse_table_path(full_table_path)
    target_table = get_target_table("profile")

    query = f"""
        SELECT column_name, data_type, mean, min_val, p25, median, p75, max_val
        FROM {target_table}
        WHERE catalog_name = '{catalog}' AND schema_name = '{schema}' AND table_name = '{table}'
    """

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
                if not rows:
                    return []
                return [ProfileColumn(**row.asDict()) for row in rows]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
