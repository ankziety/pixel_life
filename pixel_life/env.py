from __future__ import annotations

"""Pixel Life Gym environment (Phase-1 scaffold).

Design assumptions (can be revisited later):
------------------------------------------------
• Grid starts with hard boundaries; `_expand_universe` will grow the array
  when the spice agent chooses an expansion action.
• Both agents currently observe the full grid each step.  We may switch
  to windowed observations in later phases for efficiency.
• Main agent actions: each *live* pixel chooses one of five atomic actions
  (split, consume, combine, forfeit, noop).  For now we allocate a
  `MultiDiscrete` of length `H*W` and ignore entries for dead/empty cells.
• Spice agent actions: Discrete(6) → {noop, expand N, S, E, W, apply_tweak}.
• Termination: episode ends if no live pixels remain **or** we exceed
  `max_steps` (default 10_000).  Additional conditions can be added later.
"""

from typing import Dict, Set, Tuple, Optional, Any, List

import numpy as np
import gym
from gym import spaces


class PixelLifeEnv(gym.Env):
    """2-D cellular life-and-death environment with an adversarial "spice" agent."""

    metadata = {"render.modes": ["human", "ansi"]}

    # ------------------------------------------------------------------
    # Enumerations / constants
    # ------------------------------------------------------------------
    # Main-agent atomic actions (per pixel)
    ACT_SPLIT: int = 0
    ACT_CONSUME: int = 1
    ACT_COMBINE: int = 2
    ACT_FORFEIT: int = 3
    ACT_NOOP: int = 4

    NUM_MAIN_ACTIONS: int = 5

    # Spice-agent actions
    SPICE_NOOP: int = 0
    SPICE_EXPAND_N: int = 1
    SPICE_EXPAND_S: int = 2
    SPICE_EXPAND_E: int = 3
    SPICE_EXPAND_W: int = 4
    SPICE_APPLY_TWEAK: int = 5

    def __init__(
        self,
        height: int,
        width: int,
        *,
        params: Optional[Dict[str, Any]] = None,
        max_steps: int = 10_000,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        # Input validation
        if height <= 0 or width <= 0:
            raise ValueError("Grid dimensions must be positive integers")

        # Core geometry -------------------------------------------------
        self.H = height
        self.W = width
        self.origin: Tuple[int, int] = (self.H // 2, self.W // 2)

        # World state ---------------------------------------------------
        self.grid: np.ndarray = np.zeros((self.H, self.W), dtype=np.int32)
        self.organisms: Dict[int, Set[Tuple[int, int]]] = {}
        self.pixel_to_org: Dict[Tuple[int, int], int] = {}
        self._next_org_id: int = 1  # will increment after seeding

        # Parameters / timers ------------------------------------------
        self.params: Dict[str, Any] = {
            "tweak_interval": 500,
            "energy_split": 1,
            "energy_consume": 1,
            **(params or {}),
        }
        self._tweak_timer: int = 0
        self._global_step: int = 0
        self._max_steps: int = max_steps

        # RNG -----------------------------------------------------------
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Action spaces -------------------------------------------------
        max_cells = self.H * self.W  # upper bound before any expansion
        self.action_space_main = spaces.MultiDiscrete(  # type: ignore[assignment]
            np.full((max_cells,), self.NUM_MAIN_ACTIONS, dtype=np.int64)
        )
        self.action_space_spice = spaces.Discrete(6)  # 0-5 defined above

        # Observation spaces (both agents see the same grid for now) ----
        obs_shape = (self.H, self.W)
        self.observation_space_main = spaces.Box(
            low=-1, high=np.iinfo(np.int32).max, shape=obs_shape, dtype=np.int32
        )
        self.observation_space_spice = self.observation_space_main

        # Initialise state ---------------------------------------------
        self._seed_initial_organism()

    # ------------------------------------------------------------------
    # Gym API methods
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):  # type: ignore[override]
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.grid.fill(0)
        self.organisms.clear()
        self.pixel_to_org.clear()
        self._next_org_id = 1
        self._tweak_timer = 0
        self._global_step = 0
        self._seed_initial_organism()
        obs = self.grid.copy()
        info: Dict[str, Any] = {}
        return obs, info

    def step(
        self,
        spice_action: int,
        pixel_actions: np.ndarray,
    ):
        """Advance one tick.  (Full logic to be added in Phase-3.)"""
        raise NotImplementedError("Phase-3 will implement `step` logic.")

    # ------------------------------------------------------------------
    # Rendering utilities (simple ASCII)
    # ------------------------------------------------------------------
    def render(self, mode: str = "human"):
        if mode not in self.metadata["render.modes"]:
            raise ValueError(f"Unsupported mode: {mode}")
        rows: List[str] = []
        for y in range(self.H):
            row_chars: List[str] = [self._repr_cell(val) for val in self.grid[y]]
            rows.append("".join(row_chars))
        picture = "\n".join(rows)
        if mode == "human":
            print(picture)
            return None
        return picture

    # ------------------------------------------------------------------
    # Helper stubs (to implement in Phase-2)
    # ------------------------------------------------------------------
    def do_split(self, y: int, x: int) -> None:  # pragma: no cover
        raise NotImplementedError

    def do_consume(self, y: int, x: int) -> None:  # pragma: no cover
        raise NotImplementedError

    def do_combine(self, y: int, x: int, group_cells: Set[Tuple[int, int]]):  # pragma: no cover
        raise NotImplementedError

    def do_forfeit(self, y: int, x: int):  # pragma: no cover
        raise NotImplementedError

    def _expand_universe(self, direction: int):  # pragma: no cover
        raise NotImplementedError

    def _apply_tweak(self):  # pragma: no cover
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _seed_initial_organism(self):
        """Create organism ID=1 occupying the origin cell."""
        org_id = self._allocate_org_id()
        y, x = self.origin
        self.grid[y, x] = org_id
        self.organisms[org_id] = {(y, x)}
        self.pixel_to_org[(y, x)] = org_id

    def _allocate_org_id(self) -> int:
        oid = self._next_org_id
        self._next_org_id += 1
        return oid

    @staticmethod
    def _repr_cell(v: int) -> str:
        if v == 0:
            return "."
        if v == -1:
            return "#"
        if 1 <= v <= 9:
            return str(v)
        if 10 <= v < 36:
            return chr(ord("A") + v - 10)
        return "*"