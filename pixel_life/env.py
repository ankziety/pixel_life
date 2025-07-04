from __future__ import annotations

import math
from typing import Dict, Set, Tuple, Optional, Any

import numpy as np
import gym
from gym import spaces


class PixelLifeEnv(gym.Env):
    """A 2-D "pixel life" Gym environment (Phase-1 scaffold).

    Notes
    -----
    * Grid values: ``0`` = empty, ``>0`` = organism ID, ``-1`` = dead/decayed pixel.
    * For simplicity, the observation for *both* agents is currently the full grid.
    * The action spaces are placeholders and **will be refined** in later phases.
    * Helper methods (`do_split`, `do_consume`, etc.) are declared but not implemented yet.
    """

    metadata = {"render.modes": ["human", "ansi"]}

    # --- Constants ---------------------------------------------------------
    MAIN_NOOP: int = 0  # placeholder – per-pixel action ``0`` means "noop"
    SPICE_NOOP: int = 0  # spice action ``0`` means "noop"

    def __init__(
        self,
        height: int,
        width: int,
        *,
        params: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        # Grid dimensions (may change if the universe expands).
        self.H = height
        self.W = width

        # Core state holders -------------------------------------------------
        self.grid: np.ndarray = np.zeros((self.H, self.W), dtype=np.int32)
        self.organisms: Dict[int, Set[Tuple[int, int]]] = {}
        self.pixel_to_org: Dict[Tuple[int, int], int] = {}
        self.origin: Tuple[int, int] = (self.H // 2, self.W // 2)

        # Rules / parameters
        self.params: Dict[str, Any] = params.copy() if params else {}

        # Timers & bookkeeping
        self._tweak_timer: int = 0
        self._global_step: int = 0

        # --- RNG -----------------------------------------------------------
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        # -------------------------------------------------------------------
        # Action & observation spaces – very rough placeholders for now.
        #   * Main agent: one discrete action per potential live pixel up to
        #     an upper bound ``max_live`` (defaults to full grid size). This
        #     will be tightened in future phases.
        #   * Spice agent: simple Discrete space – later expanded.
        # -------------------------------------------------------------------
        max_live = self.H * self.W  # worst case upper bound
        self.NUM_MAIN_ACTIONS = 5  # split/consume/combine/forfeit/noop (tentative)
        self.action_space_main = spaces.MultiDiscrete(
            np.full((max_live,), self.NUM_MAIN_ACTIONS, dtype=np.int64)
        )
        self.action_space_spice = spaces.Discrete(6)  # noop + 4 expansions + tweak

        # Observation: full grid (H×W) integers
        obs_shape = (self.H, self.W)
        self.observation_space_main = spaces.Box(
            low=-1, high=2 ** 31 - 1, shape=obs_shape, dtype=np.int32
        )
        # Spice agent sees same grid for now
        self.observation_space_spice = self.observation_space_main

        # Seed initial organism (ID=1) at origin
        self._next_org_id = 2  # reserve 1 for initial organism
        self._seed_initial_organism()

    # ------------------------------------------------------------------
    # Standard Gym API --------------------------------------------------
    # ------------------------------------------------------------------
    def reset(self) -> Tuple[np.ndarray, dict]:  # type: ignore[override]
        """Reset environment state and return initial observation for **main** agent.

        The spice agent shares the same observation for now; differentiation can
        be added later by returning a tuple or using `self.last_obs`.
        """
        self.grid.fill(0)
        self.organisms.clear()
        self.pixel_to_org.clear()
        self._tweak_timer = 0
        self._global_step = 0
        self._next_org_id = 2
        self._seed_initial_organism()
        obs = self.grid.copy()
        info: dict[str, Any] = {}
        return obs, info

    def step(
        self,
        spice_action: int,
        pixel_actions: np.ndarray,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[float, float], bool, bool, dict]:
        """Advance one tick.  ***Not yet implemented (Phase-3).***"""
        raise NotImplementedError("`step` logic to be implemented in Phase-3")

    # ------------------------------------------------------------------
    # Rendering ---------------------------------------------------------
    # ------------------------------------------------------------------
    def render(self, mode: str = "human") -> Optional[str]:  # type: ignore[override]
        """Simple ASCII renderer (placeholder)."""
        if mode not in self.metadata["render.modes"]:
            raise ValueError(f"Unsupported render mode: {mode}")

        grid_str = "\n".join("".join(self._repr_cell(v) for v in row) for row in self.grid)
        if mode == "human":
            print(grid_str)
            return None
        return grid_str

    # ------------------------------------------------------------------
    # Helper stubs (Phase-2) -------------------------------------------
    # ------------------------------------------------------------------
    def do_split(self, y: int, x: int) -> None:
        raise NotImplementedError

    def do_consume(self, y: int, x: int) -> None:
        raise NotImplementedError

    def do_combine(self, y: int, x: int, group_cells: Set[Tuple[int, int]]) -> None:
        raise NotImplementedError

    def do_forfeit(self, y: int, x: int) -> None:
        raise NotImplementedError

    def _expand_universe(self, direction: int) -> None:
        raise NotImplementedError

    def _apply_tweak(self) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Internal utilities ------------------------------------------------
    # ------------------------------------------------------------------
    def _seed_initial_organism(self) -> None:
        """Place a single-pixel organism (ID=1) at `self.origin`."""
        y, x = self.origin
        self.grid[y, x] = 1
        self.organisms[1] = {(y, x)}
        self.pixel_to_org[(y, x)] = 1

    @staticmethod
    def _repr_cell(v: int) -> str:
        if v == 0:
            return "."
        if v == -1:
            return "#"
        # Positive: organism ID → map 1-9 then A-Z then "*" fallback
        if 1 <= v <= 9:
            return str(v)
        if 10 <= v < 36:
            return chr(ord("A") + v - 10)
        return "*"