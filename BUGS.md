# Three real bugs hit while building

Each one surfaced from a failing eval scenario. The fix and a regression test are listed for each. All three regression tests are in [tests/test_regressions.py](tests/test_regressions.py).

## 1. The trace sanitizer redacted every date as a phone number

- **What broke:** While debugging the "same electrician as last time" scenario, every slot and date in the trace showed as `[redacted-phone]` (for example `"slot": "[redacted-phone]T09:00"`). The trace stopped being useful for debugging. It also could not prove which slot was approved and which was executed.
- **Why:** The phone-number regex `\+?\d[\d\- ]{8,}\d` made the `+` optional and allowed hyphens. `2026-11-10` is digits and hyphens with more than 8 characters, so it matched.
- **Fix:** In our data, phone numbers always have a country prefix. The regex now requires it: `\+\d{1,3}[\- ]?\d{5}[\- ]?\d{3,5}\b` ([agent/trace.py](agent/trace.py)).
- **Regression test:** `test_bug1_trace_sanitizer_keeps_iso_dates` checks that ISO dates and times are kept and that `+91-…` numbers are still redacted.

## 2. The claim guard rewrote truthful "nothing was booked" messages

- **What broke:** When a turn failed, the agent replied: "I haven't booked, cancelled or changed anything. Nothing has been booked or changed." The trace showed a `claim_blocked` event for the agent's own safe error message.
- **Why:** The guard that stops the model claiming unverified success included `has been (booked|cancelled|rescheduled)`. It had no negation handling, so "Nothing **has been booked**" counted as a success claim. The bad side is a false positive: it doesn't break safety, but it garbles the user-facing messages that matter most. It would have also blocked the model from saying "your booking has not been cancelled".
- **Fix:** Added negative lookbehinds for `nothing` and `not` to the claim regex ([agent/orchestrator.py](agent/orchestrator.py)).
- **Regression test:** `test_bug2_claim_guard_ignores_negations_but_catches_claims` checks three negations that must pass and four claims that must still be caught.

## 3. Restart recovery crashed the agent on startup

- **What broke:** In the "restart mid-booking" scenario, the agent crashed after the provider had committed the booking. Starting a new agent on the same DB then failed with `TypeError: Tracer.emit() got multiple values for argument 'kind'`. The process could not start again as long as an in-flight operation existed. The recovery path meant to prevent duplicate bookings was the thing that crashed.
- **Why:** `Agent.notify(text, kind=...)` passed `kind=` through to `Tracer.emit(kind, **data)`, whose first positional parameter is also `kind`. That code path runs only during recovery or proactive notifications, which ordinary turns never reach.
- **Fix:** Renamed the parameter to `channel` (`notify(text, channel="recovery")`).
- **Regression test:** `test_bug3_recovery_notification_on_restart_does_not_crash` crashes inside `book()` after the commit, restarts on the same SQLite file, and checks that the recovery notice is delivered. The eval scenarios "Restart mid-booking, after/before the provider committed" also cover it end to end.

## Also caught by the evals

These are not counted as the three bugs above.
- **Budget grounding:** A scripted model invented a ₹1,200 budget the user never stated. The new budget-grounding check rejected it. That was the check working correctly, and the test fixture was fixed.
- **Proactive replacement:** No in-budget AC provider existed near the user. The agent correctly said "no replacement found". The seed data was tuned so the demo has a valid replacement.
