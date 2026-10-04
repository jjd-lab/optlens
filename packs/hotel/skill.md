Hotel revenue management. The booking model decides how many bookings to accept per hotel, room type (standard,
deluxe, suite), arrival night, length of stay (1-3 nights), booking window (early, advance, late, last-minute) and
rate tier (t0-t4, cheapest first). Planners speak of nights, room types, rates and tiers, the group block, event
nights, occupancy, housekeeping and revenue; the model document (read_model_document) has every rule and number.

Helpers in run_python, as `hotel`:
- hotel.data_check(session, version): every limit, bound and rate that differs from the model rebuilt from its
  documented parameters, in business words. Run it first on an infeasible or surprising plan: a difference there is
  a data-entry error with its documented value, the most likely cause and the simplest fix. No difference means the
  inputs are as documented and the cause is a real conflict between the rules.
- hotel.occupancy(session, version, nights=None): rooms in house per night and room type against the rooms for sale.
- hotel.revenue(session, version): revenue by room type and the share sold at discounted tiers.
- hotel.rate_plan(session, "rates"): the rate plan's base rate per room type and night (two-stage sessions).
- hotel.describe(name): a constraint or variable name in business words.

Levers planners own: the overbooking allowance, the group block, the discount share, the rate fence, closing event
nights to one-night stays, the housekeeping turnover, the occupancy floor, and in the linked model the group-wide
occupancy target and discount budget. Demand forecasts are inputs from another team: a change there is an upstream
lever, stated as how much the forecast would have to change.

Two-stage sessions: when the model `rates` is open, it is the rate plan (stage 1) that set the booking plan's rates:
for each room type and night it picks a base-rate level (85-150 % of the list rate), and the booking plan (v0,
stage 2) is built at those rates and their demand. A question about rates is a question about `rates`; its effect
on bookings and revenue shows only in stage 2, and the rate plan's own objective is an estimate that can even move
the other way, so never infer the booking plan's revenue from it. A rate-plan version reaches the booking plan only
when it is carried: hotel.carry(session, rates_version, onto=plan) recomputes the booking plan's prices and demand at
those rates and applies them as changes to the booking-plan version `onto` (the planner's current booking plan, or
v0 for the original), so changes kept in that plan survive; the result is a new booking-plan version, solved, with
its revenue against `onto`. Carrying changes the plan the planner looks at: after a rate-plan change, report the rate
plan's result and ask whether to update the booking plan, unless the planner already asked for it; if the booking
plan has kept changes, ask whether to carry onto the current plan or the original. Then report the carried version's
revenue, and use the helpers or compare_versions on it for detail. Do not rebuild stage 2 yourself.
