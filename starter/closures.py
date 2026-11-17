"""Closure handling for La Pedrera, computed from the ONNX input alone.

The model never receives a date, so it cannot know that tomorrow is a closure.
It can only see closures that already happened inside its 7-day input window.
This module uses that, plus the published calendar for the evaluation window
(15 Dec 2026 - 15 Jan 2027), in three steps:

1. Detect   closed_days(): a day whose afternoon load is as low as the night.
2. Clean    CleanClosedDays: closed input days are replaced by the mean profile
            of the open days, so a closure does not drag down the next week.
3. Rules    applied to the forecast, using the published calendar:
            * yesterday closed, first closed day, a Friday (25 Dec) -> reopen
            * yesterday closed otherwise (Jan maintenance)          -> stay closed
            * oldest input day is a closed Friday (25 Dec -> 1 Jan) -> opens 11:00

Use cleaning and rules together, and only at inference/export time: on the
historical closures (Mondays 6 Jul and 7 Sep) the building reopened the next
day, so training with "stay closed" would teach the wrong thing.

Usage once the team has a model:  model = ClosureAwareForecast(ConsumptionPredictor(...))
Investigation and tests:          starter/closure_analysis.ipynb
"""

from __future__ import annotations

import torch
import torch.nn as nn

WINDOW = 168
HORIZON = 24
DAYS_IN_WINDOW = WINDOW // HORIZON

CLOSED_RATIO = 1.4      # afternoon / night load below this -> closed day
FRIDAY = 4.0
NIGHT_MAX_HOUR = 5.0    # hours 0..5 = base load
DAYTIME_HOURS = (12.0, 18.0)
PRE_OPENING_HOURS = (6, 7)  # forecast positions with the lowest load of the day
LATE_OPENING_HOURS = (8, 9)  # positions held at pre-opening level on 1 Jan
RAMP_HOUR = 8           # usual ramp-up position, moved to 10:00 on 1 Jan


def closed_days(x: torch.Tensor) -> torch.Tensor:
    """1.0 for each of the 7 input days that looks closed, else 0.0. Output: [B, 7].

    The window is split into 7 consecutive 24 h blocks (block 6 = yesterday).
    """
    kwh, hour = x[..., 0], x[..., 1]
    batch = x.shape[0]

    night = (hour <= NIGHT_MAX_HOUR).float()
    daytime = ((hour >= DAYTIME_HOURS[0]) & (hour <= DAYTIME_HOURS[1])).float()
    night_level = (kwh * night).sum(1) / night.sum(1).clamp(min=1.0)           # [B]

    day_kwh = (kwh * daytime).reshape(batch, DAYS_IN_WINDOW, HORIZON).sum(2)
    day_count = daytime.reshape(batch, DAYS_IN_WINDOW, HORIZON).sum(2).clamp(min=1.0)
    daytime_level = day_kwh / day_count                                         # [B, 7]
    return (daytime_level < CLOSED_RATIO * night_level.unsqueeze(1)).float()


class ClosureFeatures(nn.Module):
    """Flags about closures visible in the input window. Output: [B, 11].

    0 closed_yesterday   last 24 h looked like a closed day
    1 closed_streak      consecutive closed days at the end of the window / 7
    2 christmas_reopen   first closed day and it was a Friday -> reopen
    3 late_opening       oldest day is a closed Friday, the rest open (1 Jan)
    4..10 closed_day_k   one flag per input day, oldest (4) to yesterday (10)

    Can also be concatenated to a network's inputs as extra features.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dow = x[..., 2]
        closed = closed_days(x)                                                 # [B, 7]

        # Count closed days backwards from yesterday until the first open day.
        # (Unrolled loop: torch.cumprod has no ONNX opset 17 export.)
        still_closed = torch.ones_like(closed[:, 0])
        streak = torch.zeros_like(closed[:, 0])                                 # [B]
        for block in range(DAYS_IN_WINDOW - 1, -1, -1):
            still_closed = still_closed * closed[:, block]
            streak = streak + still_closed

        closed_yesterday = closed[:, -1]
        christmas_reopen = closed_yesterday * (streak == 1).float() * (dow[:, -1] == FRIDAY).float()
        late_opening = (
            closed[:, 0] * (dow[:, 0] == FRIDAY).float() * (closed[:, 1:].sum(1) == 0).float()
        )
        flags = torch.stack(
            [closed_yesterday, streak / DAYS_IN_WINDOW, christmas_reopen, late_opening], dim=1
        )
        return torch.cat([flags, closed], dim=1)


class CleanClosedDays(nn.Module):
    """Replace the kWh of closed input days with the mean profile of the open days.

    Hour and day-of-week channels are untouched. If every day is closed the
    input is returned unchanged (there is no open day to copy from).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        closed = closed_days(x).unsqueeze(2)                                    # [B, 7, 1]
        days = x[..., 0].reshape(batch, DAYS_IN_WINDOW, HORIZON)                # [B, 7, 24]
        open_count = (1.0 - closed).sum(1)                                      # [B, 1]
        open_profile = ((1.0 - closed) * days).sum(1) / open_count.clamp(min=1.0)  # [B, 24]

        has_open_day = (open_count > 0).float().unsqueeze(1)                    # [B, 1, 1]
        replace = closed * has_open_day
        cleaned = replace * open_profile.unsqueeze(1) + (1.0 - replace) * days
        return torch.cat([cleaned.reshape(batch, WINDOW, 1), x[..., 1:]], dim=2)


class ClosureAwareForecast(nn.Module):
    """Wrap a forecaster: clean its input and override its output with the rules.

    The base model sees the cleaned input; the rules always read the raw input,
    so "stay closed" still works during a multi-day closure.
    """

    def __init__(self, base: nn.Module, clean_input: bool = True, rules: bool = True):
        super().__init__()
        self.base = base
        self.clean = CleanClosedDays() if clean_input else nn.Identity()
        self.rules = rules
        self.features = ClosureFeatures()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(self.clean(x))
        if not self.rules:
            return y
        flags = self.features(x)
        closed_yesterday, christmas_reopen, late_opening = flags[:, 0], flags[:, 2], flags[:, 3]

        # Stay closed: repeat yesterday (a closed day) for the whole horizon.
        stay_closed = (closed_yesterday * (1.0 - christmas_reopen)).unsqueeze(1)
        y = stay_closed * x[:, -HORIZON:, 0] + (1.0 - stay_closed) * y

        # Late opening on 1 Jan: hold 08:00-09:00 at the pre-opening level and
        # move the usual 08:00 ramp to 10:00.
        pre_opening = y[:, list(PRE_OPENING_HOURS)].mean(1, keepdim=True)
        late_y = torch.cat(
            [
                y[:, : LATE_OPENING_HOURS[0]],
                pre_opening.expand(-1, len(LATE_OPENING_HOURS)),
                y[:, RAMP_HOUR : RAMP_HOUR + 1],
                y[:, LATE_OPENING_HOURS[-1] + 2 :],
            ],
            dim=1,
        )
        late = late_opening.unsqueeze(1)
        return late * late_y + (1.0 - late) * y
