from typing import Any

from tinfer.core.request import AlignmentType, AlignmentItem, Alignment


class AlignmentFormatter:
    def __init__(self) -> None:
        pass

    def to_websocket_format(
        self,
        alignment: Alignment,
        normalized: bool = False,
    ) -> dict[str, Any]:
        if not alignment.items:
            return {"chars": [], "charStartTimesMs": [], "charDurationsMs": []}
        
        chars = []
        char_start_times = []
        char_durations = []
        
        for align in alignment.items:
            chars.append(align.item)
            char_start_times.append(align.start_ms)
            char_durations.append(align.end_ms - align.start_ms)
        
        return {
            "chars": chars,
            "charStartTimesMs": char_start_times,
            "charDurationsMs": char_durations,
        }


    def to_grpc_format(self, alignment: Alignment) -> dict[str, Any]:
        return [
            {
                "word": align.item,
                "start_ms": align.start_ms,
                "end_ms": align.end_ms,
            } for align in alignment.items
        ]