"""The mock tools honour the fixed contracts and failure behaviours from the assignment."""
from collections import Counter

import pytest

from mocks.tools import MockWorld, Scenario, ToolTimeout, ToolUnavailable
from mocks.world_seed import INJECTION_REVIEWS


def _hold(w, pid="plum_00"):
    slot = w.search_providers("plumber", "Koramangala", {"start": "2026-11-10", "end": "2026-11-15"})
    p = next(x for x in slot["providers"] if x["provider_id"] == pid)
    return w.hold_slot(pid, p["available_slots"][0])


def test_search_returns_3_to_8_providers_with_required_fields():
    w = MockWorld()
    for svc in ("plumber", "electrician", "ac_service"):
        r = w.search_providers(svc, "Koramangala", {"start": "2026-11-10", "end": "2026-11-15"})["providers"]
        assert 3 <= len(r) <= 8
        assert all({"provider_id", "price", "rating", "available_slots", "attributes"} <= set(p) for p in r)


def test_get_provider_has_prompt_injection_review():
    assert INJECTION_REVIEWS[0] in MockWorld().get_provider("plum_00")["reviews"]


def test_hold_lasts_10_minutes_and_price_drift_configurable():
    w = MockWorld(Scenario(hold_price_delta={"plum_00": 300}))
    h = _hold(w)
    assert h["quoted_price"] == w.catalog["plum_00"]["price"] + 300
    w.clock.advance(minutes=10)
    assert w.book(h["hold_id"], "k1") == {"error": "hold_expired"}


def test_book_outcome_distribution_is_70_15_15_with_half_of_timeouts_committed():
    counts = Counter()
    for seed in range(2000):
        w = MockWorld(Scenario(seed=seed))
        h = _hold(w)
        try:
            r = w.book(h["hold_id"], f"k{seed}")
            counts["success" if "booking_id" in r else r["error"]] += 1
        except ToolTimeout:
            counts["timeout_committed" if w.get_booking_status(idempotency_key=f"k{seed}")["status"] == "confirmed"
                   else "timeout_lost"] += 1
    n = sum(counts.values())
    assert abs(counts["success"] / n - 0.70) < 0.04
    assert abs(counts["slot_taken"] / n - 0.15) < 0.03
    assert abs((counts["timeout_committed"] + counts["timeout_lost"]) / n - 0.15) < 0.03
    assert abs(counts["timeout_committed"] - counts["timeout_lost"]) / n < 0.03


def test_book_is_idempotent_on_key():
    w = MockWorld(Scenario(book_outcomes=["success", "success"]))
    h = _hold(w)
    assert w.book(h["hold_id"], "same") == w.book(h["hold_id"], "same")
    assert w.count_confirmed("plum_00") == 1


def test_cancel_first_call_503_then_succeeds():
    w = MockWorld()
    with pytest.raises(ToolUnavailable):
        w.cancel("bk_up_ac01", "c1")
    assert w.cancel("bk_up_ac01", "c1")["status"] == "cancelled"


def test_reschedule_slot_conflict_on_configured_slots():
    w = MockWorld(Scenario(conflict_slots=["2026-11-14T10:00"]))
    assert w.reschedule("bk_up_ac01", "2026-11-14T10:00", "r1") == {"error": "slot_conflict"}
    assert w.reschedule("bk_up_ac01", "2026-11-14T11:00", "r2")["slot"] == "2026-11-14T11:00"


def test_list_bookings_shape():
    b = MockWorld().list_bookings("u1")["bookings"]
    assert {"booking_id", "provider_id", "service", "slot", "status"} <= set(b[0])
