"""Offline stand-in for Claude (CAD_ASSIST_AI_REPLAY=1): runs the loop end to end with no API key or cost.

It starts from the best-fitting template, tries thinner gussets to chase the mass target, and backs off if the
validator objects. Useful for exercising the UI; it is not a designer.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace as NS

IN = 25.4


class ReplayClient:
    def __init__(self, delay: float = 0.8):
        self.delay = delay
        self.turn = 0
        self.template: list[dict] | None = None
        self.builds = 0

    def create(self, **kw):
        time.sleep(self.delay)
        self.turn += 1
        last = kw["messages"][-1]["content"]
        result = last[0]["content"] if isinstance(last, list) else ""
        usage = NS(input_tokens=0, output_tokens=0, cache_read_input_tokens=0)

        def reply(*blocks):
            return NS(content=list(blocks), stop_reason="tool_use", usage=usage)

        def tool(name, args):
            return NS(type="tool_use", id=f"r{self.turn}", name=name, input=args)

        def text(t):
            return NS(type="text", text=t)

        if self.turn == 1:
            state = json.loads(last.split("(mm):\n", 1)[1])
            if state["current_design"]:
                self.template = state["current_design"]
            else:
                forked = any(o["kind"] == "bore" and o.get("gap") for o in state["intent"])
                return reply(tool("get_template", {"name": "single_lug_fitting" if forked else "clevis_bracket"}))
        if self.template is None:
            self.template = json.loads(result)
        self.builds += 1
        if self.builds == 1:
            return reply(text("Replay mode: building the starting design as is."), tool("build_and_check", {"note": "Baseline", "features": self.template}))
        if self.builds == 2:
            lighter = json.loads(json.dumps(self.template))
            for f in lighter:
                if f["kind"] == "gusset_pair":
                    f["params"]["thickness"] = 0.12 * IN
            return reply(tool("build_and_check", {"note": "Thinner gussets (0.12 in) to save mass", "features": lighter}))
        return reply(tool("finish", {"summary": "Replay run: kept the better of the baseline and the thinner-gusset variant. Real designs need the API key."}))
