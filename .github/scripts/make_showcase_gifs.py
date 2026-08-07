#!/usr/bin/env python
"""
Generate the showcase GIFs used by the README and the docs landing page.

Four scenarios of increasing complexity are rendered, all with 10% obstacles
and **untrained, uniformly random** actions -- these clips illustrate what the
environment can be configured to express, not learned behaviour:

  * ``showcase_1v1_10x10.gif``      1 predator vs 1 prey, 10x10
  * ``showcase_2v2_10x10.gif``      2 predators vs 2 prey at double speed, 10x10
  * ``showcase_3v3v10_50x50.gif``   two predator teams with different movement
                                    geometries (cardinal vs diagonal) vs 10 prey, 50x50
  * ``showcase_5v20_100x100.gif``   5 predators vs 20 prey, 100x100

Everything runs offscreen (Pillow, no display, no pygame window), so the script
is headless- and CI-safe. Colours and marker shapes come from the environment's
own ``Agent.get_agent_color()`` / ``Agent._shape_for_subteam()``, and movement
geometry from the shipped ``DiscreteActionSpace`` / ``CrossActionSpace``
plugins, so the clips show the same visual language and the same dynamics a
user gets from ``plug-and-play/scripts/render.py``.

Three things worth knowing about how these clips are composed:

1. **Double speed** (scenario 2) is the real ``SpeedWrapper`` mechanic: a
   logical step is replayed as ``min(speed, stamina)`` sub-steps, so the prey
   advance two cells per frame. Stamina is sized per scenario
   (``Scenario.stamina_for``) rather than left at the shipped default of 10,
   which would otherwise strand an agent motionless partway through the clip.
   Episodes end at the first capture and the clip resets and keeps filming,
   because the core freezes captured prey in place and keeps drawing them.
2. **Per-team movement geometry** (scenario 3) is *not* a supported core
   feature -- ``GridWorldEnv`` holds one global ``action_space_plugin``. The
   clip composes it through the core's documented per-agent fallback: with no
   plugin attached, ``step()`` reads each agent's own
   ``_actions_to_directions`` map, which is populated here from the real
   plugin classes. It illustrates a capability the wrapper-layer roadmap
   (Tier 1) would make configurable; do not read it as YAML-selectable today.
3. Two legibility-driven deviations from the pygame renderer, confined to the
   large grids where a faithful copy is unreadable at collage size: grid lines
   fade to light grey (pygame draws solid black), and markers are scaled up
   (pygame uses exactly ``pix / 3``). Per-agent name labels are omitted
   throughout for the same reason.

Run from the repository root (needs ``pip install -e ".[docs]"``)::

    python .github/scripts/make_showcase_gifs.py

Each clip is written twice, mirroring how ``demo.gif`` is already handled:
``miscellenous/gifs/showcase/`` for the README (GitHub resolves README paths
from the repository root) and ``docs/assets/images/showcase/`` for the docs
site (MkDocs can only publish files under ``docs/``). Both copies are
byte-identical and committed, so the README and docs render without
regenerating them; re-run this script whenever rendering conventions change.
"""

from __future__ import annotations

import os

# Headless: no display is opened, but Agent's constructor initialises
# pygame.font, so pin a dummy video driver before that import happens.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import math  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, List, Optional, Tuple  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from ppage.actions.cross_actions import CrossActionSpace  # noqa: E402
from ppage.actions.discrete_actions import DiscreteActionSpace  # noqa: E402
from ppage.core.agent import Agent  # noqa: E402
from ppage.core.gridworld import GridWorldEnv  # noqa: E402
from ppage.wrappers.speed import SpeedWrapper  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

# Each clip is written to both locations, following the convention demo.gif
# already uses: the README (served from the repository root on GitHub) reads
# the miscellenous/ copy, while the docs site can only publish files that live
# under docs/. Writing both from one script keeps them from drifting apart.
OUTPUT_DIRS = [
    REPO_ROOT / "miscellenous" / "gifs" / "showcase",
    REPO_ROOT / "docs" / "assets" / "images" / "showcase",
]

OBSTACLE_PERCENT = 10.0
BACKGROUND = (255, 255, 255)
OBSTACLE_COLOR = (50, 50, 50)
FRAME_MS = 160
N_ACTIONS = 5
PALETTE_COLORS = 32

# Movement geometries, taken from the shipped action-space plugins so the
# direction vectors are exactly those a configured experiment would use.
GEOMETRIES = {
    "plus": DiscreteActionSpace(),  # cardinal: right / up / left / down / noop
    "cross": CrossActionSpace(),  # diagonal: NE / NW / SW / SE / noop
}


@dataclass
class Group:
    """A block of identically configured agents sharing one team label."""

    agent_type: str
    team: str
    count: int
    speed: int = 1
    geometry: Optional[str] = None  # None -> the Agent's own cardinal map


@dataclass
class Scenario:
    """One showcase clip: its roster, grid, and render settings."""

    slug: str
    size: int
    canvas: int
    frames: int
    seed: int
    groups: List[Group]
    total_subteams: Optional[int] = None
    marker_scale: float = 1.0
    # On dense grids, ring each marker in white so agents separate from the
    # obstacle field instead of reading as more clutter.
    halo: bool = False
    caption: str = ""
    stats: Dict[str, int] = field(default_factory=dict)

    @property
    def per_agent_geometry(self) -> bool:
        """
        True when any group names a movement geometry.

        The per-agent direction map is only consulted while no global action
        plugin is attached, so naming *any* geometry means the plugin has to
        stay off. Keying this off "more than one distinct geometry" would
        silently ignore a single named geometry.
        """
        return any(g.geometry for g in self.groups)

    @property
    def max_speed(self) -> int:
        return max((g.speed for g in self.groups), default=1)

    def stamina_for(self, group: "Group") -> int:
        """
        Stamina large enough that no agent can run dry inside the clip.

        SpeedWrapper spends one stamina per sub-step and only refills on
        reset(), so the shipped default of 10 would freeze an agent partway
        through and misrepresent the mechanic being demonstrated.
        """
        return (self.frames + 2) * max(1, group.speed)


SCENARIOS = [
    Scenario(
        slug="showcase_1v1_10x10",
        size=10,
        canvas=400,
        frames=55,
        seed=11,
        groups=[
            Group("predator", "predator_1", 1),
            Group("prey", "prey_1", 1),
        ],
        caption="1 predator vs 1 prey, 10x10",
    ),
    Scenario(
        slug="showcase_2v2_10x10",
        size=10,
        canvas=400,
        frames=55,
        seed=23,
        groups=[
            Group("predator", "predator_1", 2),
            # Double speed via SpeedWrapper. Stamina is sized per scenario (see
            # Scenario.stamina_for) so neither side runs dry mid-clip.
            Group("prey", "prey_1", 2, speed=2),
        ],
        caption="2 predators vs 2 prey at double speed, 10x10",
    ),
    Scenario(
        slug="showcase_3v3v10_50x50",
        size=50,
        canvas=560,
        frames=60,
        seed=31,
        groups=[
            # Deep red squares, cardinal movement. Subteam 2 is the darker of
            # the two red shades the colour function produces.
            Group("predator", "predator_2", 3, geometry="plus"),
            # Pale pink circles, diagonal-only movement.
            Group("predator", "predator_1", 3, geometry="cross"),
            Group("prey", "prey_1", 10, geometry="plus"),
        ],
        # Two predator subteams exist, so tell the colour function that: it
        # spreads saturation/value across `total_subteams`, separating the teams
        # into a deep and a pale red instead of two near-identical tints.
        total_subteams=2,
        marker_scale=1.9,
        halo=True,
        caption=(
            "Two predator teams, cardinal (red) vs diagonal (pink) movement, "
            "vs 10 prey, 50x50"
        ),
    ),
    Scenario(
        slug="showcase_5v20_100x100",
        size=100,
        canvas=700,
        frames=60,
        seed=47,
        groups=[
            Group("predator", "predator_1", 5),
            Group("prey", "prey_1", 20),
        ],
        marker_scale=2.5,
        halo=True,
        caption="5 predators vs 20 prey, 100x100",
    ),
]


def build_agents(scn: Scenario) -> List[Agent]:
    """Expand the roster into configured, individually named Agent instances."""
    agents: List[Agent] = []
    for group in scn.groups:
        for i in range(1, group.count + 1):
            agent = Agent(
                agent_type=group.agent_type,
                agent_team=group.team,
                agent_name=f"{group.team}_{i}",
            )
            agent.agent_speed = group.speed
            agent.stamina = scn.stamina_for(group)
            if scn.total_subteams is not None:
                agent.total_subteams = scn.total_subteams
            if group.geometry is not None:
                # Core's per-agent fallback map, used when no global plugin is
                # attached. Populated from the real plugin's direction vectors.
                plugin = GEOMETRIES[group.geometry]
                agent._actions_to_directions = {
                    a: plugin.to_direction(a) for a in range(plugin.n_actions)
                }
            agents.append(agent)

    names = [a.agent_name for a in agents]
    if len(set(names)) != len(names):
        # Agent names key the action dict and the wrapper's speed/stamina
        # tables, so a collision would silently drop an agent's action.
        raise ValueError(f"{scn.slug}: duplicate agent names in roster: {names}")
    return agents


def make_env(scn: Scenario, agents: List[Agent]):
    """Build the env for a scenario; returns (steppable, core_env)."""
    env = GridWorldEnv(
        agents=agents,
        size=scn.size,
        perc_num_obstacle=OBSTACLE_PERCENT,
        render_mode=None,  # frames are drawn here, offscreen
        seed=scn.seed,
        # End the episode on the first capture. The core freezes captured prey
        # in place and keeps drawing them, so letting the episode run on would
        # park a motionless agent on screen for most of the clip; build_clip
        # resets instead and films the next episode.
        capture_threshold=1,
        # SpeedWrapper replays a logical step as several core steps, and
        # max_steps counts core steps, so scale the budget by the top speed or
        # the clip truncates early.
        max_steps=(scn.frames + 1) * scn.max_speed,
    )

    if scn.per_agent_geometry:
        # Per-agent direction maps only take effect while no global plugin is
        # attached, which also rules out SpeedWrapper (it needs a plugin to
        # detect NOOP actions).
        if scn.max_speed > 1:
            raise ValueError(
                f"{scn.slug}: per-agent movement geometry cannot be combined "
                "with speeds above 1 (SpeedWrapper requires a global action "
                "plugin, which overrides the per-agent maps)"
            )
        env.action_space_plugin = None
        return env, env

    env.action_space_plugin = DiscreteActionSpace()
    if scn.max_speed > 1:
        return SpeedWrapper(env), env
    return env, env


def _star_points(
    cx: float, cy: float, outer: float, inner: float, points: int = 5
) -> List[Tuple[float, float]]:
    """Mirror of Agent._star_points, in float space for smoother output."""
    pts: List[Tuple[float, float]] = []
    step = math.pi / points
    angle = -math.pi / 2
    for i in range(points * 2):
        r = outer if i % 2 == 0 else inner
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
        angle += step
    return pts


def _draw_marker(
    draw: ImageDraw.ImageDraw,
    shape: str,
    cx: float,
    cy: float,
    r: float,
    color: Tuple[int, int, int],
) -> None:
    """Draw one agent marker, matching the shapes of Agent._draw_agent."""
    if shape == "square":
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif shape == "triangle":
        draw.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=color)
    elif shape == "diamond":
        draw.polygon(
            [(cx, cy - r), (cx - r, cy), (cx, cy + r), (cx + r, cy)], fill=color
        )
    elif shape == "star":
        draw.polygon(_star_points(cx, cy, r, r * 0.45), fill=color)
    else:  # "circle", and any unknown shape rather than drawing nothing
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)


def render_frame(env: GridWorldEnv, scn: Scenario) -> Image.Image:
    """Render the environment's current state to an RGB image."""
    pix = scn.canvas / scn.size
    img = Image.new("RGB", (scn.canvas, scn.canvas), BACKGROUND)
    draw = ImageDraw.Draw(img)

    # obstacles: dark squares of side pix/2, as in the pygame renderer
    obstacle_r = max(1.0, pix / 4)
    for obs in env._obstacle_location:
        cx = (float(obs[0]) + 0.5) * pix
        cy = (float(obs[1]) + 0.5) * pix
        draw.rectangle(
            [cx - obstacle_r, cy - obstacle_r, cx + obstacle_r, cy + obstacle_r],
            fill=OBSTACLE_COLOR,
        )

    # grid lines: solid black in pygame; faded here once cells get small
    if pix >= 20:
        line_color = (0, 0, 0)
    elif pix >= 10:
        line_color = (205, 205, 205)
    else:
        line_color = (232, 232, 232)
    for i in range(scn.size + 1):
        pos = round(pix * i)
        draw.line([(0, pos), (scn.canvas, pos)], fill=line_color, width=1)
        draw.line([(pos, 0), (pos, scn.canvas)], fill=line_color, width=1)

    for agent in env.agents:
        _, sub_id = agent._parse_team()
        shape = agent._shape_for_subteam(sub_id)
        cx = (float(agent._agent_location[0]) + 0.5) * pix
        cy = (float(agent._agent_location[1]) + 0.5) * pix
        r = max(2.0, (pix / 3) * scn.marker_scale)
        if scn.halo:
            _draw_marker(draw, shape, cx, cy, r + max(1.5, r * 0.45), BACKGROUND)
        _draw_marker(draw, shape, cx, cy, r, agent.get_agent_color())

    return img


def build_clip(scn: Scenario) -> List[Image.Image]:
    """Roll out one scenario under random actions, returning its frames."""
    agents = build_agents(scn)
    steppable, core = make_env(scn, agents)
    steppable.reset(seed=scn.seed)

    # Policy randomness is kept separate from the environment's own generator,
    # so obstacle layout and spawn positions depend only on the scenario seed.
    policy_rng = np.random.default_rng(scn.seed + 1000)

    frames = [render_frame(core, scn)]
    episodes = 1
    captures = 0
    for _ in range(scn.frames):
        actions = {
            ag.agent_name: int(policy_rng.integers(N_ACTIONS)) for ag in core.agents
        }
        result = steppable.step(actions)
        frames.append(render_frame(core, scn))
        if result["terminated"] or result["truncated"]:
            # Show the terminal frame briefly, then start a fresh episode so the
            # clip keeps moving rather than holding on frozen captured prey.
            # reset() without a seed lets the environment's generator carry on,
            # giving a new layout while the whole clip stays reproducible.
            captures += len(core._captured_agents)
            frames.append(frames[-1])
            steppable.reset()
            episodes += 1
            frames.append(render_frame(core, scn))

    frames.extend([frames[-1]] * 3)  # brief hold on the final frame
    scn.stats = {
        "agents": len(core.agents),
        "obstacles": len(core._obstacle_location),
        "episodes": episodes,
        "captures": captures,
        "rendered": len(frames),
    }
    steppable.close()
    return frames


def save_gif(frames: List[Image.Image], path: Path) -> int:
    """Write frames as an optimised, looping GIF; returns its size in bytes."""
    # One shared palette across every frame: it keeps colours stable (no
    # flicker) and lets the GIF writer emit only the changed region per frame,
    # which matters on the large, mostly-static grids.
    base = frames[0].convert("P", palette=Image.Palette.ADAPTIVE, colors=PALETTE_COLORS)
    quantised = [base] + [
        f.quantize(palette=base, dither=Image.Dither.NONE) for f in frames[1:]
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    quantised[0].save(
        path,
        save_all=True,
        append_images=quantised[1:],
        duration=FRAME_MS,
        loop=0,
        optimize=True,
    )
    return path.stat().st_size


def main() -> None:
    total = 0
    for scn in SCENARIOS:
        frames = build_clip(scn)
        primary, *mirrors = OUTPUT_DIRS
        size = save_gif(frames, primary / f"{scn.slug}.gif")
        for mirror in mirrors:
            # Copy the encoded bytes rather than re-encoding, so every
            # destination holds a byte-identical file.
            target = mirror / f"{scn.slug}.gif"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((primary / f"{scn.slug}.gif").read_bytes())
        total += size
        # Report the frame count actually present in the written file: the GIF
        # writer drops frames identical to their predecessor, so the number of
        # rendered frames overstates it.
        with Image.open(primary / f"{scn.slug}.gif") as written:
            n_frames = getattr(written, "n_frames", 1)
        print(
            f"wrote {scn.slug}.gif  "
            f"{n_frames} frames (of {scn.stats['rendered']} rendered)  "
            f"{scn.stats['agents']} agents  "
            f"{scn.stats['obstacles']} obstacles  "
            f"{scn.stats['episodes']} episodes  "
            f"{scn.stats['captures']} captures  "
            f"{size / 1024:.0f} KB"
        )
    print(f"total {total / 1024:.0f} KB in {len(OUTPUT_DIRS)} locations")
    for d in OUTPUT_DIRS:
        print(f"  -> {d.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
