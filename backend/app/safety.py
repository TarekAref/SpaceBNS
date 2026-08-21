"""Shared safety envelope and global exception handlers for SpaceBNS.

The four mandatory safety-envelope fields are defined once here and injected
into every response path — normal, degraded, and error — so that they can
never diverge between endpoints.

No model calls, no file I/O, no global state mutation.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

# ---------------------------------------------------------------------------
# Shared safety envelope (contract Section 10)
# ---------------------------------------------------------------------------

SAFETY_ENVELOPE: dict[str, str] = {
    "data_source": "SYNTHETIC",
    "prototype_status": "NOT_FLIGHT_QUALIFIED",
    "command_authority": "NONE",
    "policy_decision": "PERMITTED_FOR_SIMULATION_ONLY",
}


# ---------------------------------------------------------------------------
# Handler registration
# ---------------------------------------------------------------------------

def register_exception_handlers(app: FastAPI) -> None:
    """Register all required exception handlers on *app*.

    Every handled path merges the safety envelope into the response body so
    that ``command_authority: "NONE"`` is always present.  Stack traces and
    filesystem paths are never exposed.
    """

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        body = dict(SAFETY_ENVELOPE)
        # Only accept plain string detail values as public error codes.
        # Never stringify dict/exception bodies — that could leak internals.
        if isinstance(exc.detail, str):
            body["error"] = exc.detail
        else:
            body["error"] = "REQUEST_ERROR"
        return JSONResponse(status_code=exc.status_code, content=body)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        body = dict(SAFETY_ENVELOPE)
        # Inspect only Pydantic error type and location to detect missing sample
        # fields.  Never include msg, input, ctx, exception text, paths, or
        # stack traces.  All other validation failures return VALIDATION_ERROR.
        #
        # FastAPI wraps Pydantic v2 errors in RequestValidationError whose
        # .errors() returns a plain list of dicts (no kwargs accepted).
        # For a missing field in samples[i], loc is a 4-tuple/list:
        #   ("body", "samples", <index as str or int>, "<field_name>")
        try:
            for err in exc.errors():
                err_type = err.get("type", "")
                loc = err.get("loc", ())
                # POST {} — samples key entirely absent:
                # loc = ("body", "samples") — length 2
                if (
                    err_type == "missing"
                    and len(loc) == 2
                    and loc[0] == "body"
                    and loc[1] == "samples"
                ):
                    body["error"] = "EMPTY_WINDOW"
                    return JSONResponse(status_code=422, content=body)
                # Missing field within a specific sample:
                # loc = ("body", "samples", <index>, "<field_name>") — length 4
                if (
                    err_type == "missing"
                    and len(loc) == 4
                    and loc[0] == "body"
                    and loc[1] == "samples"
                    and isinstance(loc[3], str)
                ):
                    # loc[2] may be an int or a str depending on FastAPI version
                    try:
                        sample_index = int(loc[2])
                    except (ValueError, TypeError):
                        continue
                    body["error"] = "INVALID_SAMPLE_SCHEMA"
                    body["sample_index"] = sample_index
                    body["field"] = loc[3]
                    return JSONResponse(status_code=422, content=body)
        except Exception:  # noqa: BLE001
            pass
        body["error"] = "VALIDATION_ERROR"
        # Do not include exc.errors() details — the ctx dict may contain
        # non-JSON-serialisable Python exception objects and could expose
        # internal details.
        return JSONResponse(status_code=422, content=body)

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        body = dict(SAFETY_ENVELOPE)
        body["error"] = "INTERNAL_SERVER_ERROR"
        # Never expose type name, message, or traceback — just a fixed string.
        body["message"] = "An unexpected error occurred."
        return JSONResponse(status_code=500, content=body)
