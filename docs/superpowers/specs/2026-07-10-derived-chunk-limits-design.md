# Derived Chunk Limits Design

## Goal

Separate immediate generation from mandatory splitting without adding another
configuration field. Derive both decisions from `chunk_length_schedule` while
keeping early inference chunks small and avoiding tiny remainder chunks.

## Derived limits

For chunk index `i`, the current schedule value is the generation trigger and
preferred split target. The next schedule value is the inclusive no-split
limit.

Given `[120, 160, 250, 290]`, chunk index zero uses:

- trigger and preferred split target: `120`
- no-split limit: `160`

Pending text up to 120 characters waits for timeout or force. Pending text from
121 through 160 characters generates immediately as one chunk. Pending text
above 160 characters generates immediately and is split around the 120-character
target.

## Final schedule entry

When there is no next entry, extrapolate the no-split limit using the most
recent schedule increase. For `[120, 160, 250, 290]`, the final limit is
`290 + (290 - 250) = 330`.

For a one-entry schedule, add one third of that value as headroom. A schedule of
`[120]` therefore derives a no-split limit of `160`. Derived headroom is always
at least one character.

## Splitting and continuations

When splitting is required, existing sentence-aware splitting continues to use
the current schedule entry as its preferred maximum. Prepared chunks from the
same pending-text snapshot continue immediately after the preceding inference
finishes. Text appended after that snapshot remains outside the prepared queue
and follows normal trigger timing.

This look-ahead rule turns a 125-character initial snapshot into one inference,
while 161 characters split to approximately 120 and 41 characters. It avoids
the former 120-and-5 result without allowing the first chunk to grow beyond the
next scheduled size.

## API and verification

No public configuration fields are added or removed. Documentation will define
each schedule value as a generation trigger and preferred split target, with the
next value serving as derived no-split headroom.

Do not add tests. Verify with existing checks and direct probes at the exact
trigger, trigger plus one, no-split limit, and no-split limit plus one for
multi-entry, final-entry, and single-entry schedules.
