# Hotel revenue management


## Model Overview

This optimization model plans room sales for a hotel group over a booking horizon. For every hotel, room type,
arrival night, length of stay, booking window and rate tier, it decides how many bookings to accept, so that total
room revenue is as high as possible. Demand is forecast separately for each rate tier (a cheaper tier sells more
rooms), and the hotel decides how much of each tier to sell: the classic booking-limit model of hotel revenue
management. Key business rules: a room type cannot hold more guests on a night than it has rooms, plus a small
overbooking allowance and minus the rooms held for a group contract; discounted tiers may fill only part of a
night's arrivals; last-minute bookers never get the discounted tiers; event nights are closed to one-night stays;
housekeeping can only turn over so many rooms a day; and every night must reach a minimum occupancy.

All data is synthetic (seeded by `generate.py`); the default instance is one hotel over 14 nights. The same model
runs for any number of nights and hotels.


## Model Components

### Indices
- **h**: Hotel (`h1`, `h2`, ...). Each hotel has the same room types and rules; its market is 1.0, 1.15, 1.3 or 1.45 times
  the first hotel's (repeating every four hotels).
- **r**: Room type: `standard`, `deluxe`, `suite`.
- **s**: Length of stay in nights: `1n`, `2n`, `3n`. A booking arriving on night d with length s occupies nights
  d, ..., d+s-1 and checks out on the morning of night d+s.
- **d**: Arrival night, `d0` to `d{N-1}`. Nights 4 and 5 of each week (Friday and Saturday) have higher demand.
  Night 9 and every 30th night after it are event nights.
- **w**: Booking window: `early` (60 or more days before arrival), `advance` (14 to 59), `late` (3 to 13),
  `lastminute` (0 to 2).
- **t**: Rate tier `t0` to `t4`, cheapest first.

### Input Data
- **room_count[r]**: Rooms of each type per hotel: standard 120, deluxe 60, suite 20 (200 in all).
- **overbooking[r]**: Share of rooms that may be sold above physical capacity, because some guests do not show up:
  standard 5 %, deluxe 3 %, suite 0 %.
- **group_block[r]**: Rooms held every night for a corporate group contract and not for sale: standard 10, deluxe
  0, suite 0.
- **base_rate[r]**: Published nightly rate: standard $140, deluxe $210, suite $420.
- **tier_multiplier[t]**: Each tier's rate as a share of the base rate: 0.80, 0.90, 1.00, 1.15, 1.30. The nightly
  rate of tier t is base_rate[r] × tier_multiplier[t] (standard: $112, $126, $140, $161, $182).
- **stay_discount[s]**: Per-night discount for longer stays: 1 night 0 %, 2 nights 5 %, 3 nights 10 %.
- **demand[h,r,s,d,w,t]**: Forecast bookings available at each tier, rounded down to whole rooms. It is the
  room type's base arrivals (standard 75, deluxe 32, suite 9 a night) × the length-of-stay share (50 %, 30 %, 20 %) ×
  the booking-window share (early 20 %, advance 35 %, late 30 %, last-minute 15 %) × the tier's demand factor
  (1.45, 1.20, 1.00, 0.78, 0.60) × the night's lift (weekend × 1.3, event × 1.6, a yearly season of ±25 %, and up
  to ±10 % day-to-day noise).
- **discount_share**: Discounted tiers (t0 and t1) may fill at most 35 % of a room type's rooms in arrivals per
  night.
- **housekeeping_share**: Check-outs a day may not exceed 55 % of all rooms (110), the turnover the housekeeping
  team can clean.
- **occupancy_floor**: At least 40 % of all rooms (80, of which the group block covers 10) must be occupied every
  night, a brand standard.

### Decision Variables
- **sell[h,r,s,d,w,t]**: Non-negative integer. Bookings accepted for hotel h, room type r, a stay of s nights
  arriving on night d, booked in window w at rate tier t. Business interpretation: the booking limit the revenue
  manager loads into the reservation system for that rate.

### Objective
Maximize total room revenue: the sum over all bookings of sell × nightly rate × (1 − stay discount) × nights
stayed.

### Constraints
- **demand_limit[h,r,s,d,w,t]**: sell ≤ demand. A tier cannot sell more bookings than the forecast says will book
  at that rate.
- **room_capacity[h,r,d]**: Bookings of room type r in house on night d (arrivals on nights d−s+1 to d with length
  s) ≤ room_count × (1 + overbooking), rounded down, − group_block. Standard: 126 − 10 = 116; deluxe 61; suite 20.
- **discount_cap[h,r,d]**: Bookings arriving on night d at tiers t0 and t1, all lengths and windows, ≤
  discount_share × room_count (standard 42, deluxe 21, suite 7).
- **rate_fence[h,r,d]**: Last-minute bookings at tiers t0 and t1 ≤ 0. Business meaning: discounts reward booking
  ahead; a guest booking two days out pays at least the base tier.
- **closed_to_one_night[h,r,d]** (event nights only): One-night arrivals on an event night ≤ 0. Business meaning:
  on event nights the hotel takes only stays of two nights or more.
- **housekeeping[h,d]**: Check-outs on the morning of night d (bookings that arrived on night d−s with length s,
  all room types) ≤ housekeeping_share × total rooms = 110.
- **occupancy_floor[h,d]**: Rooms in house on night d across all room types ≥ occupancy_floor × total rooms −
  total group block = 80 − 10 = 70.

### Size
About 190 rows and 180 integer columns per hotel-night: 14 nights is about 2,700 rows; 365 nights about 69,000;
12 hotels over 365 nights about 830,000.

### Linked variant (group-wide rows)
Some instances (`generate.py --linked`) add two group-wide rules across all hotels. With them the hotels share a
budget, so the model no longer splits into one independent plan per hotel.
- **group_occupancy[d]**: Rooms in house on night d across every hotel and room type ≥ group_occupancy_floor ×
  total rooms of the group, with group_occupancy_floor = 60 %: 0.6 × 200 × number of hotels (4 hotels: 480; 12
  hotels: 1,440).
- **group_discount_budget**: The revenue the group gives up by selling below the base tier, summed over every hotel,
  night, room type, stay length and window, ≤ discount_budget × number of hotels × number of nights, with
  discount_budget = $300 per hotel-night (4 hotels × 365 nights: 438,000; 12 hotels × 365 nights: 1,314,000). A
  booking's discount is (base rate − tier rate) × (1 − stay discount) × nights: per night, standard $28 at t0 and $14
  at t1, deluxe $42 and $21, suite $84 and $42. Tiers t2 and above give no discount.

### Two-stage variant (rate plan, then bookings)
Revenue management first sets each night's rates (stage 1, the rate plan), and the booking model (stage 2, the model
above) then decides how many bookings to accept at those rates. The rate plan's choices are the booking model's
data: a different rate changes the booking model's revenue per booking and its demand. `generate.py --pipeline DIR`
writes `rates.lp`, solves it, and writes `levels.json` (the chosen levels), `bookings.lp` (stage 2 at those rates)
and `params.json`.

**Stage 1, rate plan** (`rates.lp`, a small MILP: 15 binaries and about 12 rows per hotel-night)
- **rate_level[h,r,d,L]** (binary): room type r at hotel h on night d is sold at base-rate level L. The levels are
  85 %, 100 %, 115 %, 130 % and 150 % of the list base rate (standard $140, deluxe $210, suite $420); a level's
  arrivals are 120 %, 100 %, 84 %, 70 % and 56 % of the list-rate arrivals (higher rates sell fewer rooms).
- **Objective, expected_revenue**: maximize the sum of base rate × expected rooms, where expected rooms at a level
  are the night's arrivals at that level × an average stay of 1.7 nights, capped by the rooms of that type.
- **one_rate[h,r,d]**: exactly one level per hotel, room type and night.
- **room_type_ladder[h,r,d]**: deluxe's base rate ≥ standard's + $40, and suite's ≥ deluxe's + $120, every night.
- **rate_step_up[h,r,d]** and **rate_step_down[h,r,d]**: a base rate moves at most 30 % of the list rate from one
  night to the next (standard $42, deluxe $63, suite $126).
- **event_premium[h,r,d]**: on event nights the base rate is at least 130 % of the list rate (standard $182, deluxe
  $273, suite $546).
- **occupancy_promise[h,d]**: the expected rooms sold on each night ≥ 55 % of all rooms (110 of 200).

**Stage 2, bookings** (`bookings.lp`): the booking model above, with every tier's rate scaled by the night's chosen
level (tier rate = base rate × level × tier multiplier) and every demand limit scaled by that level's arrivals. With
the list rates (level 100 % everywhere) it is exactly the one-stage model.
