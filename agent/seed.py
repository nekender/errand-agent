"""Seeds the agent DB with 50+ stored memories and a long prior conversation (for context-budget tests
and the demo)."""
from __future__ import annotations

from agent.memory import Memory

# (key, value, scope, source, confirmed)
CORE_FACTS = [
    ("preferred_area", "Koramangala", "global", "explicit", True),
    ("earliest_hour", "9", "global", "explicit", True),
    ("contact_preference", "WhatsApp", "global", "explicit", True),
    ("access_notes", "4th floor, lift available, visitor parking near gate 2", "global", "explicit", True),
    ("pets", "Has a dog; please keep the main door closed", "global", "explicit", True),
    ("payment_preference", "UPI after the job", "global", "explicit", True),
    ("preferred_provider", "elec_02 (Ravi Kumar)", "electrician", "explicit", True),
    ("budget_max", "1500", "plumber", "explicit", True),
    ("ac_details", "2 Daikin split ACs, bedroom unit drips", "ac_service", "explicit", True),
    ("cleaning_notes", "Bring own supplies; no bleach", "cleaning", "explicit", True),
    # Inferred from history — NOT eligible for context until the user confirms.
    ("prefers_weekends", "true", "global", "inferred", False),
    ("latest_hour", "18", "global", "inferred", False),
    ("preferred_provider", "plum_04", "plumber", "inferred", False),
    # Inferred and confirmed by the user — eligible.
    ("language", "English", "global", "inferred", True),
]
FORGOTTEN = [
    ("old_area", "Whitefield", "global"),
    ("preferred_provider", "clea_03", "cleaning"),
]
# Low-relevance household trivia so the context packer has to choose.
FILLER = [
    ("favourite_cricket_team", "RCB"), ("society_name", "Prestige Garden Apartments"),
    ("wifi_router_location", "living room shelf"), ("kids_school_time", "7:30 AM pickup"),
    ("water_tank_cleaning", "last done in July"), ("inverter_brand", "Luminous"),
    ("geyser_age", "6 years"), ("fridge_model", "LG double door"), ("washing_machine", "front load"),
    ("balcony_plants", "12 pots"), ("diet", "vegetarian"), ("guest_room", "converted to office"),
    ("work_from_home_days", "Mon, Wed, Fri"), ("meeting_hours", "11 AM - 1 PM weekdays"),
    ("doorbell", "broken, please call"), ("security_guard_name", "Ramesh"),
    ("garbage_pickup", "8 AM daily"), ("maid_schedule", "7 AM - 9 AM"), ("car", "white hatchback"),
    ("bike_parking", "basement B2"), ("intercom", "flat 4B"), ("landlord_contact", "via WhatsApp only"),
    ("lease_renewal", "March"), ("paint_colour", "off-white"), ("sofa_cleaning", "due in December"),
    ("mattress_size", "king"), ("tv_mount", "living room wall"), ("water_purifier", "RO, service yearly"),
    ("gas_cylinder", "booked via app"), ("gym_time", "6 PM"), ("music_preference", "quiet during calls"),
    ("allergy", "dust"), ("preferred_language_for_calls", "Hindi or English"), ("newspaper", "Deccan Herald"),
    ("milk_delivery", "6 AM"), ("courier_drop", "leave with security"), ("solar_panel", "none"),
    ("smart_lock", "code shared on request"),
]


def seed_memories(memory: Memory) -> dict:
    ids = {"core": [], "forgotten": [], "filler": []}
    for key, value, scope, source, confirmed in CORE_FACTS:
        ids["core"].append(memory.add(key, value, source=source, scope=scope, confirmed=confirmed))
    for key, value, scope in FORGOTTEN:
        fid = memory.add(key, value, source="explicit", scope=scope)
        memory.forget(fact_id=fid)
        ids["forgotten"].append(fid)
    for key, value in FILLER:
        ids["filler"].append(memory.add(key, value, source="explicit", scope="global"))
    return ids


PRIOR_TURNS = [
    ("user", "Can you find someone to fix the bathroom geyser?"),
    ("assistant", "Sure — which day works, and what's your budget?"),
    ("user", "Saturday morning, under 1000."),
    ("assistant", "I found 3 electricians; Ravi Kumar (4.7★, ₹850) has 10 AM free. Shall I propose it?"),
    ("user", "Yes, Ravi is great."),
    ("assistant", "Done earlier: Ravi Kumar visited on 14 Oct at 10 AM."),
    ("user", "Also remind me the plumber last time was okay but late."),
    ("assistant", "Noted."),
]


def seed_conversation(db, user_id: str, session_id: str = "prior", repeats: int = 25) -> int:
    n = 0
    for i in range(repeats):
        for role, text in PRIOR_TURNS:
            db.execute("INSERT INTO messages(session_id,user_id,role,content,created_at) VALUES(?,?,?,?,?)",
                       (session_id, user_id, role, f"{text} (prior #{i})", "2026-10-01T10:00:00"))
            n += 1
    return n
