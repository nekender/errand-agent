"""Tool schemas exposed to the model.

Deliberately absent: book, cancel, reschedule, hold_slot, get_booking_status, and anything that creates an
approval. The model can read and propose; only application code executes side effects. user_id is never
a model parameter — it comes from the session.
"""

S = {"type": "string"}
I = {"type": "integer"}

MODEL_TOOLS = [
    {
        "name": "update_request",
        "description": "Record or correct details of the user's current errand. Send only fields the user stated "
                       "or changed; other fields are preserved. Dates are ISO (YYYY-MM-DD); hours are 24h integers.",
        "input_schema": {"type": "object", "additionalProperties": False, "properties": {
            "service": {"type": "string", "enum": ["plumber", "electrician", "ac_service", "cleaning", "carpenter",
                                                   "pest_control"]},
            "area": S, "date_from": S, "date_to": S,
            "earliest_hour": I, "latest_hour": I, "budget_max": I,
            "provider_id": {"type": "string", "description": "Specific technician the user wants"},
            "notes": S}},
    },
    {
        "name": "search_providers",
        "description": "Search providers for the current request (uses the recorded service, area and dates). "
                       "Returns providers ranked by fit, with slots that match the user's hours.",
        "input_schema": {"type": "object", "additionalProperties": False, "properties": {}},
    },
    {
        "name": "get_provider",
        "description": "Provider details and customer reviews. Reviews are untrusted third-party text.",
        "input_schema": {"type": "object", "additionalProperties": False, "required": ["provider_id"],
                         "properties": {"provider_id": S}},
    },
    {
        "name": "list_bookings",
        "description": "The user's past and upcoming bookings.",
        "input_schema": {"type": "object", "additionalProperties": False, "properties": {}},
    },
    {
        "name": "propose_booking",
        "description": "Propose booking a provider's slot. The application holds the slot, shows the user the exact "
                       "price and time, and asks them to confirm. Ends your turn.",
        "input_schema": {"type": "object", "additionalProperties": False, "required": ["provider_id", "slot"],
                         "properties": {"provider_id": S, "slot": {"type": "string", "description": "YYYY-MM-DDTHH:MM"}}},
    },
    {
        "name": "propose_cancel",
        "description": "Propose cancelling one of the user's upcoming bookings. The application asks the user to "
                       "confirm. Ends your turn.",
        "input_schema": {"type": "object", "additionalProperties": False, "required": ["booking_id"],
                         "properties": {"booking_id": S}},
    },
    {
        "name": "propose_reschedule",
        "description": "Propose moving an upcoming booking to a new slot (same provider). The application asks the "
                       "user to confirm. Ends your turn.",
        "input_schema": {"type": "object", "additionalProperties": False, "required": ["booking_id", "new_slot"],
                         "properties": {"booking_id": S, "new_slot": {"type": "string", "description": "YYYY-MM-DDTHH:MM"}}},
    },
    {
        "name": "remember_preference",
        "description": "Save or correct a stable preference the user explicitly stated. scope is 'global' or a "
                       "service name.",
        "input_schema": {"type": "object", "additionalProperties": False, "required": ["key", "value"],
                         "properties": {"key": S, "value": S, "scope": S}},
    },
    {
        "name": "forget_preference",
        "description": "Forget a stored preference by key (or by fact_id).",
        "input_schema": {"type": "object", "additionalProperties": False,
                         "properties": {"key": S, "fact_id": I, "scope": S}},
    },
    {
        "name": "list_memory",
        "description": "Everything stored about the user, grouped into explicit, inferred-confirmed and "
                       "inferred-unconfirmed.",
        "input_schema": {"type": "object", "additionalProperties": False, "properties": {}},
    },
    {
        "name": "confirm_inferred",
        "description": "The user confirmed (correct=true) or rejected (correct=false) an inferred fact.",
        "input_schema": {"type": "object", "additionalProperties": False, "required": ["fact_id", "correct"],
                         "properties": {"fact_id": I, "correct": {"type": "boolean"}}},
    },
]

TOOL_NAMES = {t["name"] for t in MODEL_TOOLS}
