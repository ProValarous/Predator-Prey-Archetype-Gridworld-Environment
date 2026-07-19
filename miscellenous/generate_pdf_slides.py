#!/usr/bin/env python3
"""
Generate a polished PDF slide deck for the Predator-Prey MARL Gridworld project.
Output: slides/presentation.pdf
"""
from pathlib import Path
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, Color
from reportlab.lib.utils import simpleSplit
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── Page dimensions (16:9) ────────────────────────────────────────────────────
PW = 13.33 * inch
PH =  7.50 * inch

# ── Palette ───────────────────────────────────────────────────────────────────
NAVY   = HexColor('#1E3A5F')
TEAL   = HexColor('#0D6E6E')
WHITE  = HexColor('#FFFFFF')
LGRAY  = HexColor('#F0F0F0')
MGRAY  = HexColor('#888888')
DGRAY  = HexColor('#2D2D2D')
ORANGE = HexColor('#E67E22')
RED    = HexColor('#C0392B')
GREEN  = HexColor('#1A7A40')
BLUE   = HexColor('#216CB4')
PURPLE = HexColor('#6C3B9E')
AMBER  = HexColor('#F19C12')
CODEBG = HexColor('#E8E8E8')
L1     = HexColor('#C0392B')   # Layer 1 core
L2     = HexColor('#E67E22')   # Layer 2 plugins
L3     = HexColor('#1A7A40')   # Layer 3 baselines
L4     = HexColor('#216CB4')   # Layer 4 scripts
LGREEN = HexColor('#EBF7EE')
LRED   = HexColor('#FDF0EF')
LBLUE  = HexColor('#EEF4FB')
LPURP  = HexColor('#F3EEF9')

# ── Canvas ────────────────────────────────────────────────────────────────────
OUT = Path(__file__).parent.parent / 'slides' / 'presentation.pdf'
OUT.parent.mkdir(exist_ok=True)
c = Canvas(str(OUT), pagesize=(PW, PH))

slide_num = [0]

# ── Coordinate helpers (top-left origin, inches) ─────────────────────────────
def px(x_in):  return x_in * inch
def py(y_in):  return PH - y_in * inch          # top → rl bottom
def pw(w_in):  return w_in * inch
def ph(h_in):  return h_in * inch

# ── Drawing primitives ────────────────────────────────────────────────────────

def fill_page(color):
    c.setFillColor(color)
    c.rect(0, 0, PW, PH, fill=1, stroke=0)

def box(l, t, w, h, fill=None, stroke=None, sw=1):
    if fill:
        c.setFillColor(fill)
    if stroke:
        c.setStrokeColor(stroke)
        c.setLineWidth(sw)
    c.rect(px(l), py(t+h), pw(w), ph(h),
           fill=1 if fill else 0,
           stroke=1 if stroke else 0)

def hline(l, t, w, color, lw=1):
    c.setStrokeColor(color)
    c.setLineWidth(lw)
    c.line(px(l), py(t), px(l+w), py(t))

def txt(text, l, t, size=14, color=DGRAY, bold=False, align='left',
        font=None):
    """Draw a single line of text."""
    if font is None:
        font = 'Helvetica-Bold' if bold else 'Helvetica'
    c.setFont(font, size)
    c.setFillColor(color)
    x = px(l)
    y = py(t) - size * 0.75  # baseline offset
    if align == 'center':
        # centre within 'l' treated as centre x
        c.drawCentredString(x, y, text)
    elif align == 'right':
        c.drawRightString(x, y, text)
    else:
        c.drawString(x, y, text)

def wrapped_txt(text, l, t, w, h, size=13, color=DGRAY, bold=False,
                line_gap=None, font=None, align='left'):
    """Draw word-wrapped text within a box (top-left origin, inches)."""
    if font is None:
        font = 'Helvetica-Bold' if bold else 'Helvetica'
    if line_gap is None:
        line_gap = size * 1.35
    c.setFont(font, size)
    c.setFillColor(color)
    max_w = pw(w)
    lines = []
    for para in text.split('\n'):
        wrapped = simpleSplit(para, font, size, max_w)
        lines.extend(wrapped if wrapped else [''])
    rl_x  = px(l)
    rl_y0 = py(t) - size * 0.85
    for i, line in enumerate(lines):
        rl_y = rl_y0 - i * line_gap
        if rl_y < py(t + h):
            break
        if align == 'center':
            c.drawCentredString(rl_x + max_w/2, rl_y, line)
        else:
            c.drawString(rl_x, rl_y, line)

def code_block(src, l, t, w, h, size=10):
    """Draw a gray code block."""
    box(l, t, w, h, fill=CODEBG)
    font = 'Courier'
    c.setFont(font, size)
    c.setFillColor(DGRAY)
    max_w  = pw(w - 0.24)
    rl_x   = px(l + 0.12)
    rl_y0  = py(t + 0.1) - size * 0.85
    lh     = size * 1.3
    for i, line in enumerate(src.split('\n')):
        rl_y = rl_y0 - i * lh
        if rl_y < py(t + h - 0.05):
            break
        c.drawString(rl_x, rl_y, line)

def slide_header(title, hcolor=NAVY, tcolor=WHITE, hh=0.82):
    box(0, 0, 13.33, hh, fill=hcolor)
    # orange accent strip at very top
    box(0, 0, 13.33, 0.06, fill=ORANGE)
    wrapped_txt(title, 0.35, 0.13, 12.6, hh-0.15,
                size=22, color=tcolor, bold=True)

def new_slide(title=None, hcolor=NAVY, bg=WHITE):
    slide_num[0] += 1
    c.showPage()
    fill_page(bg)
    if title:
        slide_header(title, hcolor=hcolor)

def section_slide(title, subtitle=''):
    slide_num[0] += 1
    c.showPage()
    fill_page(NAVY)
    # accent bar
    box(1.0, 3.3, 11.33, 0.07, fill=ORANGE)
    wrapped_txt(title, 0.5, 1.85, 12.33, 1.4,
                size=36, color=WHITE, bold=True, align='center')
    if subtitle:
        wrapped_txt(subtitle, 0.5, 3.55, 12.33, 0.8,
                    size=18, color=HexColor('#B0C4DE'), align='center')

def bullet_row(items, l, t, w, size=15, gap=0.38, color=DGRAY):
    """Draw a list of bullet strings starting at (l, t)."""
    y_cur = t
    for item in items:
        if isinstance(item, str):
            s, lvl = item, 0
        else:
            s, lvl = item[0], item[1]
        sym  = '•  ' if lvl == 0 else '◦  '
        pad  = 0.28 * lvl
        sz   = size if lvl == 0 else size - 1
        wrapped_txt(sym + s, l + pad, y_cur, w - pad, gap * 1.5,
                    size=sz, color=color)
        y_cur += gap
    return y_cur

def simple_table(data, l, t, w, h,
                 col_fracs=None, hfill=NAVY, htxt=WHITE,
                 fs=12, alt=True):
    """Draw a simple table with header row."""
    rows = len(data)
    cols = len(data[0])
    col_w = [(w * f) for f in (col_fracs or [1/cols]*cols)]
    row_h = h / rows

    for ri, row in enumerate(data):
        x_cur = l
        for ci, cell in enumerate(row):
            cw = col_w[ci]
            fill = hfill if ri == 0 else (LGRAY if alt and ri % 2 == 0 else WHITE)
            box(x_cur, t + ri*row_h, cw, row_h, fill=fill,
                stroke=HexColor('#CCCCCC'), sw=0.5)
            fc = htxt if ri == 0 else DGRAY
            fb = (ri == 0)
            wrapped_txt(str(cell),
                        x_cur + 0.1, t + ri*row_h + 0.06,
                        cw - 0.2, row_h - 0.1,
                        size=fs, color=fc, bold=fb)
            x_cur += cw

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — Title
# ══════════════════════════════════════════════════════════════════════════════
slide_num[0] += 1
fill_page(NAVY)
box(0, 0, 13.33, 0.3, fill=ORANGE)
box(0, 7.2, 13.33, 0.3, fill=ORANGE)

wrapped_txt('Predator-Prey Gridworld',
            0.6, 1.2, 12.13, 1.4,
            size=46, color=WHITE, bold=True, align='center')
wrapped_txt('A Modular Multi-Agent Reinforcement Learning Testbed',
            0.6, 2.85, 12.13, 0.7,
            size=22, color=HexColor('#B0C4DE'), align='center')

hline(2.5, 3.75, 8.33, ORANGE, lw=1.5)

wrapped_txt('Student Orientation  •  Week 1',
            0.6, 4.05, 12.13, 0.5,
            size=17, color=HexColor('#90A8C8'), align='center')
wrapped_txt('Predator-Prey Archetype Gridworld Environment',
            0.6, 4.65, 12.13, 0.5,
            size=13, color=MGRAY, align='center')

# ══════════════════════════════════════════════════════════════════════════════
# SECTION — Why This Exists
# ══════════════════════════════════════════════════════════════════════════════
section_slide('Section 1: Why This Exists',
              'The problem we solve and our mission')

# ── Slide 2 — The Problem ────────────────────────────────────────────────────
new_slide('The Problem with Most MARL Environments')
wrapped_txt('Most research environments tangle concerns that should stay independent:',
            0.5, 0.95, 12.33, 0.42, size=15, color=DGRAY)
bullet_row([
    'Environment dynamics and learning algorithm live in the same class',
    'Opaque reward computation — you cannot trace why a policy improves',
    'Global random state makes results non-reproducible across machines',
    'Changing one variable (e.g. reward shape) silently affects others',
    'Impossible to run clean ablations: observation vs reward vs algorithm',
], 0.55, 1.48, 12.23, size=16, gap=0.44)

box(0.5, 5.08, 12.33, 1.65, fill=LBLUE, stroke=BLUE, sw=1)
wrapped_txt(
    '“We need an environment where each layer can be studied independently, '
    'with full reproducibility and zero surprise interactions between components.”',
    0.82, 5.22, 11.7, 1.38, size=15, color=NAVY)

# ── Slide 3 — Is / Is Not ────────────────────────────────────────────────────
new_slide('What This Project Is — and Is Not')

box(0.4, 1.0, 5.9, 5.75, fill=LGREEN, stroke=GREEN, sw=1)
txt('IS', 0.72, 1.08, size=20, color=GREEN, bold=True)
y_ = 1.62
for s in [
    'A controlled lab for teaching and researching RL/MARL',
    'Config-driven — one YAML change = one experimental variable',
    'Fully deterministic — same seed, same trajectory, always',
    'Plugin-based — swap observations, rewards, algorithms',
    'Traceable — every transition verifiable by hand',
]:
    wrapped_txt('✓  ' + s, 0.65, y_, 5.4, 0.55, size=14, color=DGRAY)
    y_ += 0.68

box(7.03, 1.0, 5.9, 5.75, fill=LRED, stroke=RED, sw=1)
txt('IS NOT', 7.35, 1.08, size=20, color=RED, bold=True)
y_ = 1.62
for s in [
    'A high-performance training platform',
    'A continuous-action or visual environment',
    'A replacement for PettingZoo or SMAC',
    'Optimised for GPU-scale deep RL',
    'A ready-made publication benchmark',
]:
    wrapped_txt('✗  ' + s, 7.28, y_, 5.4, 0.55, size=14, color=DGRAY)
    y_ += 0.68

# ══════════════════════════════════════════════════════════════════════════════
# SECTION — Architecture
# ══════════════════════════════════════════════════════════════════════════════
section_slide('Section 2: Architecture', 'Four layers, one golden rule')

# ── Slide 4 — Four-Layer Diagram ─────────────────────────────────────────────
new_slide('The Four-Layer Architecture')
wrapped_txt('Every component lives in exactly one layer. Dependencies flow downward only.',
            0.5, 0.92, 12.33, 0.4, size=13, color=MGRAY)

layers = [
    (L4, 'Layer 4  —  Scripts & Orchestration',
     'run_from_config.py  •  evaluate.py  •  sweep.py  •  render.py',
     'Drives training runs from YAML config'),
    (L3, 'Layer 3  —  Baselines (Learning Algorithms)',
     'IQL  •  CQL  •  MixedTrainer  •  (future: DQN, PPO, Nash Q)',
     'Selects actions, updates value tables'),
    (L2, 'Layer 2  —  Plugins (Observations + Rewards)',
     '5 observation builders  •  3 reward functions',
     'Transforms env state into percepts and signals'),
    (L1, 'Layer 1  —  Core Environment   ★ IMMUTABLE ★',
     'core/gridworld.py  •  core/agent.py',
     'Grid physics, movement, capture — never modified'),
]
y_start = 1.42
lh = 1.2
for i, (col, title, sub, note) in enumerate(layers):
    box(0.4, y_start + i*lh, 12.5, lh - 0.06, fill=col)
    wrapped_txt(title, 0.68, y_start + i*lh + 0.1,
                8.6, 0.48, size=16, color=WHITE, bold=True)
    wrapped_txt(sub,   0.68, y_start + i*lh + 0.6,
                8.6, 0.38, size=12, color=HexColor('#DDDDDD'))
    # right note box
    lighter = HexColor('#{:02X}{:02X}{:02X}'.format(
        min(255, int(col.red*255)+55),
        min(255, int(col.green*255)+55),
        min(255, int(col.blue*255)+55)))
    box(9.1, y_start + i*lh + 0.1, 3.6, lh - 0.28, fill=lighter)
    wrapped_txt(note, 9.22, y_start + i*lh + 0.22,
                3.4, 0.75, size=12, color=WHITE)

# ── Slide 5 — Design Invariants ──────────────────────────────────────────────
new_slide('Five Core Design Invariants')
wrapped_txt('These rules protect the scientific validity of every experiment run on this codebase.',
            0.5, 0.92, 12.33, 0.4, size=13, color=MGRAY)

invariants = [
    ('Immutable Core',
     'src/core/ is never modified by contributors. Physics must not change between experiments.'),
    ('Seeded Randomness',
     'env.reset(seed=n) with the same n always produces the exact same trajectory, on any machine.'),
    ('Read-Only Plugins',
     'Observation builders and reward functions observe state — they never write it.'),
    ('Config Drives Everything',
     'Need a behavioral change? Write a plugin. If you think you need to edit core code, that is a design bug.'),
    ('Black-Box Algorithms',
     'Algorithms may only call env.reset() and env.step(). No reading of env.agents or internal state.'),
]
y_ = 1.42
for title, body in invariants:
    box(0.4, y_, 12.5, 0.88, fill=LGRAY, stroke=HexColor('#CCCCCC'), sw=0.5)
    wrapped_txt(title, 0.65, y_ + 0.1, 3.2, 0.5, size=14, color=NAVY, bold=True)
    wrapped_txt(body,  3.9,  y_ + 0.12, 8.8, 0.7, size=13, color=DGRAY)
    y_ += 0.96

# ── Slide 6 — Plugin & Registry ──────────────────────────────────────────────
new_slide('The Plugin & Registry Pattern')
wrapped_txt('Plugins are discovered at runtime from YAML keys — no hardcoded imports in core.',
            0.5, 0.92, 12.33, 0.4, size=13, color=MGRAY)

flow_boxes = [
    (0.4,  'YAML Key',         'type: local_radius',           BLUE),
    (3.0,  'Registry',         'get_observation_\nbuilder(key)', TEAL),
    (5.6,  'Instantiation',    'LocalRadius\nObs(**params)',    GREEN),
    (8.2,  'Injection',        'env.observation_\nbuilder = fn', ORANGE),
    (10.8, 'Called Each Step', 'self.observation_\nbuilder(self)', RED),
]
for bx, title, body, col in flow_boxes:
    box(bx, 1.48, 2.2, 1.6, fill=col)
    wrapped_txt(title, bx+0.1, 1.53, 2.0, 0.45,
                size=12, color=WHITE, bold=True, align='center')
    wrapped_txt(body,  bx+0.1, 2.0,  2.0, 1.0,
                size=10, color=WHITE, align='center', font='Courier')
for xi in [2.62, 5.22, 7.82, 10.42]:
    txt('→', xi + 0.05, 2.12, size=22, color=NAVY, bold=True)

box(0.4, 3.32, 12.5, 3.4, fill=LBLUE, stroke=BLUE, sw=1)
wrapped_txt('Why this pattern?', 0.65, 3.42, 5.0, 0.42, size=15, color=NAVY, bold=True)
bullet_row([
    'Adding a new observation builder: one new file + one registry line — zero core changes',
    'Algorithm author never imports the builder directly — YAML is the only coupling point',
    'Core never changes — all prior experiments remain valid after adding a plugin',
    'The pattern is identical across all three extension points: obs / reward / algorithm',
], 0.65, 3.92, 12.0, size=14, gap=0.4)

# ── Slide 7 — Extension Points Table ─────────────────────────────────────────
new_slide('Extension Points — The Three Contribution Pathways')
simple_table([
    ['Layer',       'Interface Method',                       'Directory',          'Registry',                  'YAML Key'],
    ['Observation', 'ObservationBuilder.build(env) → dict', 'src/observations/', 'observation_registry.py', 'observation.type'],
    ['Reward',      'RewardFunction.compute(env) → dict',   'src/rewards/',      'reward_registry.py',      'rewards[].name'],
    ['Algorithm',   'BaseAlgorithm.select_actions()\n.train()',  'src/baselines/',    'algorithm_registry.py',   'experiment\n.algorithm.name'],
], 0.4, 1.1, 12.5, 5.6,
   col_fracs=[0.12, 0.27, 0.19, 0.22, 0.20], fs=13)
wrapped_txt('All three pathways follow the identical discover-by-string-key pattern. '
            'Contributors only ever touch one pathway.',
            0.5, 6.85, 12.33, 0.42, size=12, color=MGRAY)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION — Workflows
# ══════════════════════════════════════════════════════════════════════════════
section_slide('Section 3: Workflows',
              'Init, step-loop, and training loop in detail')

# ── Slide 8 — Initialization Flow ────────────────────────────────────────────
new_slide('Initialization Flow — From YAML to Running Environment')
wrapped_txt('What happens between   python run_from_config.py   and the first   env.step() ?',
            0.5, 0.92, 12.33, 0.4, size=13, color=MGRAY)

init_steps = [
    (BLUE,   '① Load Configs',      'load_all_configs()\nreads 5 YAML files'),
    (TEAL,   '② Build Agents',      'build_agents(cfg)\n→ list[Agent]'),
    (GREEN,  '③ Create Env',        'GridWorldEnv\n(agents, **params)'),
    (ORANGE, '④ Wire Observations', 'obs_builder\nbound as callable'),
    (PURPLE, '⑤ Wire Rewards',      'reward_fn closure\nbound to env'),
    (RED,    '⑥ Start Training',    'AlgorithmClass\n(env, cfg).train()'),
]
bw, bh = 2.0, 1.35
for i, (col, title, body) in enumerate(init_steps):
    row, ci = divmod(i, 3)
    xl = 0.35 + ci * (12.7 / 3)
    yt = 1.45 + row * (bh + 0.28)
    box(xl, yt, bw, bh, fill=col)
    wrapped_txt(title, xl+0.12, yt+0.1,  bw-0.24, 0.42, size=14, color=WHITE, bold=True)
    wrapped_txt(body,  xl+0.12, yt+0.58, bw-0.24, 0.72, size=11, color=WHITE, font='Courier')

box(0.4, 5.52, 12.5, 0.65, fill=NAVY)
wrapped_txt('Result: a fully wired environment — observations, rewards, and algorithm connected without the algorithm knowing how.',
            0.65, 5.6, 12.0, 0.5, size=14, color=WHITE, bold=True, align='center')

# ── Slide 9 — Per-Step Pipeline ──────────────────────────────────────────────
new_slide('Per-Step Pipeline — Inside  env.step(actions)')
wrapped_txt('Called once per timestep. Every stage is deterministic given the same state and seed.',
            0.5, 0.92, 12.33, 0.4, size=13, color=MGRAY)

steps9 = [
    '①  Reset per-step trackers          ( _captured_this_step = [] )',
    '②  Move all agents simultaneously   ( based on actions dict )',
    '③  Detect captures                  ( predator + prey on same cell → capture )',
    '④  Compute rewards                  ( reward_fn(self) → dict[name → float] )',
    '⑤  Check termination / truncation   ( all prey captured  OR  max_steps reached )',
    '⑥  Render frame                     ( only if render_mode = "human" or "rgb_array" )',
    '⑦  Build observations               ( observation_builder(self) → dict[name → dict] )',
    '⑧  Return result dict               ( obs, reward, terminated, truncated, info )',
]
y_ = 1.42
for i, s in enumerate(steps9):
    fill = LGRAY if i % 2 == 0 else WHITE
    box(0.4, y_, 12.5, 0.54, fill=fill, stroke=HexColor('#CCCCCC'), sw=0.5)
    wrapped_txt(s, 0.65, y_ + 0.1, 12.0, 0.38, size=14, color=DGRAY)
    y_ += 0.55
box(0.4, y_ + 0.03, 12.5, 0.52, fill=NAVY)
wrapped_txt('⚠  Returns a dict — not the standard Gymnasium (obs, rew, term, trunc, info) tuple. '
            'Gymnasium wrapper is a Week-1 deliverable.',
            0.65, y_ + 0.1, 12.0, 0.4, size=13, color=AMBER)

# ── Slide 10 — Training Loop ─────────────────────────────────────────────────
new_slide('The Training Loop — IQL Episode Flow')
code_block('''\
for episode in range(num_episodes):
    obs  = env.reset()
    done = False
    while not done:
        actions = {}
        for agent, agent_obs in obs.items():
            state = _encode_state(agent_obs)
            if random() < epsilon:
                actions[agent] = randint(0, action_dim - 1)
            else:
                actions[agent] = argmax(q_tables[agent][state])

        result = env.step(actions)

        for agent in agent_names:
            s  = _encode_state(obs[agent])
            a  = actions[agent]
            r  = result["reward"][agent]
            s_ = _encode_state(result["obs"][agent])
            td = r + gamma * max(q_tables[agent][s_].values()) \\
                   - q_tables[agent][s][a]
            q_tables[agent][s][a] += alpha * td

        obs  = result["obs"]
        done = result["terminated"] or result["truncated"]

    epsilon = max(min_epsilon, epsilon * epsilon_decay)
''', 0.4, 1.05, 7.1, 6.15, size=11)

annots = [
    (BLUE,   'Epsilon-greedy',   'Random with prob ε,\ngreedy otherwise.\nε decays per episode.'),
    (TEAL,   'Bellman Update',   'Q(s,a) += α[γ·maxQ(s’,·) − Q(s,a) + r]\nTD error drives convergence.'),
    (GREEN,  'defaultdict',      'Q-tables auto-init new\nstates to zero.\nNo pre-allocation needed.'),
    (ORANGE, 'Epsilon Decay',    'Exploration → exploitation.\nDecays after every episode,\nnot every step.'),
]
y_ = 1.1
for col, title, body in annots:
    box(7.7, y_, 5.3, 1.42, fill=col)
    wrapped_txt(title, 7.85, y_ + 0.08, 4.95, 0.42, size=14, color=WHITE, bold=True)
    wrapped_txt(body,  7.85, y_ + 0.56, 4.95, 0.82, size=12, color=WHITE)
    y_ += 1.52

# ══════════════════════════════════════════════════════════════════════════════
# SECTION — File Structure
# ══════════════════════════════════════════════════════════════════════════════
section_slide('Section 4: File Structure', 'Where everything lives and why')

# ── Slide 11 — Directory Tree ────────────────────────────────────────────────
new_slide('Annotated Directory Structure')
wrapped_txt('One directory = one responsibility.',
            0.5, 0.92, 12.33, 0.4, size=13, color=MGRAY)
code_block('''\
Predator-Prey-Archetype-Gridworld-Environment/
  src/
    multi_agent_package/
      core/           <- immutable physics (gridworld.py, agent.py)
      observations/   <- 5 builders: LocalOnly LocalRadius Default Absolute Relative
      rewards/        <- 3 functions: BaseReward PredatorDistance Survival
      registry/       <- 3 registries: observation, reward (algorithm in baselines)
      scripts/        <- run_from_config.py  evaluate.py  sweep.py  render.py
    baselines/
      IQL/  CQL/  MIXED/         <- one folder per algorithm
      base.py                    <- BaseAlgorithm abstract class
      registry/                  <- algorithm registry
  configs/           <- env.yaml  agents.yaml  observations.yaml  rewards.yaml  experiment*.yaml
  tests/             <- 189 pytest tests across 9 files (all architecture layers)
  notebooks/         <- 01_iql_demo.ipynb  02_cql_demo.ipynb
  wiki/              <- 38 documentation files (SDD, ADRs, guides, concepts)
  slides/            <- presentation.pptx  presentation.pdf
''', 0.4, 1.38, 8.7, 5.72, size=11)

box(9.35, 1.38, 3.7, 5.72, fill=LBLUE, stroke=BLUE, sw=1)
wrapped_txt('Where do I…?', 9.55, 1.45, 3.3, 0.42, size=14, color=NAVY, bold=True)
guide = [
    ('Change how agents see',  'observations/'),
    ('Change what agents want','rewards/'),
    ('Add new algorithm',      'baselines/'),
    ('Tune experiment params', 'configs/'),
    ('Understand the system',  'wiki/'),
    ('Run a demo',             'notebooks/'),
    ('Change movement rules',  'talk to maintainer'),
]
y_ = 2.0
for q, a in guide:
    wrapped_txt(q, 9.55, y_, 3.4, 0.3, size=11, color=DGRAY)
    wrapped_txt('→  ' + a, 9.55, y_+0.3, 3.4, 0.3, size=11, color=TEAL,
                bold=True, font='Courier')
    y_ += 0.68

# ── Slide 12 — Key Files Table ───────────────────────────────────────────────
new_slide('Key Files and Their Roles')
simple_table([
    ['File',                              'Layer',       'Role'],
    ['core/gridworld.py',                 'L1 Core',     'Grid simulation, step loop, capture detection'],
    ['core/agent.py',                     'L1 Core',     'Agent state, movement, 5-action map'],
    ['observations/*.py',                 'L2 Plugin',   'Transform env state to per-agent percepts'],
    ['rewards/*.py',                      'L2 Plugin',   'Compute per-agent reward signals'],
    ['registry/observation_registry.py',  'L2 Registry', 'Map YAML string keys to builder classes'],
    ['baselines/IQL/iql.py',              'L3 Baseline', 'Independent Q-Learning — one Q-table per agent'],
    ['baselines/CQL/cql.py',              'L3 Baseline', 'Centralised Q-Learning — joint action Q-tensor'],
    ['baselines/MIXED/mixed.py',          'L3 Baseline', 'Per-team algorithm assignment'],
    ['scripts/run_from_config.py',        'L4 Script',   'Entry point — loads configs, wires env, trains'],
    ['scripts/evaluate.py',               'L4 Script',   'Greedy rollout + per-agent metrics collection'],
], 0.4, 1.05, 12.5, 6.15,
   col_fracs=[0.37, 0.17, 0.46], fs=13)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION — Class & Code
# ══════════════════════════════════════════════════════════════════════════════
section_slide('Section 5: Class & Code Structure',
              'The key abstractions and how they connect')

# ── Slide 13 — Core Classes ──────────────────────────────────────────────────
new_slide('Core Classes — GridWorldEnv and Agent')
box(0.4, 1.05, 6.0, 5.95, fill=LRED, stroke=RED, sw=1)
wrapped_txt('GridWorldEnv(gym.Env)', 0.62, 1.1, 5.7, 0.48, size=16, color=RED, bold=True)
wrapped_txt('Attributes:', 0.72, 1.65, 5.5, 0.35, size=12, color=DGRAY, bold=True)
for i, a in enumerate([
    'agents              list[Agent]',
    '_obstacle_location  set of (x,y)',
    '_captured_agents    set of names',
    'reward_fn           Callable',
    'observation_builder Callable',
    'capture_threshold   int',
    'max_steps  int  |  seed  int',
]):
    wrapped_txt(a, 0.72, 2.05+i*0.34, 5.5, 0.32, size=11, color=DGRAY, font='Courier')
wrapped_txt('Methods:', 0.72, 4.45, 5.5, 0.35, size=12, color=DGRAY, bold=True)
for i, m in enumerate([
    'reset(seed) → obs_dict',
    'step(actions) → result_dict',
    'base_reward() → float',
    '_build_observations()',
]):
    wrapped_txt(m, 0.72, 4.86+i*0.34, 5.5, 0.3, size=11, color=NAVY, font='Courier')

box(7.0, 1.05, 6.0, 5.95, fill=LBLUE, stroke=BLUE, sw=1)
wrapped_txt('Agent', 7.22, 1.1, 5.7, 0.48, size=16, color=BLUE, bold=True)
wrapped_txt('Attributes:', 7.32, 1.65, 5.5, 0.35, size=12, color=DGRAY, bold=True)
for i, a in enumerate([
    'agent_name     str',
    "agent_type     'predator' | 'prey'",
    'agent_team     str',
    '_agent_location  [x, y]',
    'is_captured    bool',
    'agent_speed    int   (not yet wired)',
    'stamina        float (not yet wired)',
]):
    wrapped_txt(a, 7.32, 2.05+i*0.36, 5.5, 0.32, size=11, color=DGRAY, font='Courier')
wrapped_txt('Action Map:', 7.32, 4.62, 5.5, 0.35, size=12, color=DGRAY, bold=True)
wrapped_txt('0=Right  1=Up  2=Left  3=Down  4=Noop',
            7.32, 5.02, 5.5, 0.32, size=11, color=NAVY, font='Courier')
wrapped_txt('All agents move simultaneously — not turn-by-turn.',
            7.32, 5.46, 5.5, 0.38, size=11, color=MGRAY)

# ── Slide 14 — Plugin Interfaces ─────────────────────────────────────────────
new_slide('Plugin Interfaces — Three Abstract Base Classes')
for i, (col, title, src, note) in enumerate([
    (GREEN,  'ObservationBuilder',
     'class ObservationBuilder:\n  def __init__(self,\n               **params): ...\n\n  def build(\n      self, env\n  ) -> Dict[str, dict]:\n      # Read env state\n      # Return per-agent\n      # observation dicts\n      # READ-ONLY\n      ...',
     'env.observation_builder =\n  builder_instance.build\nCalled each step.'),
    (ORANGE, 'RewardFunction',
     'class RewardFunction:\n  def __init__(self,\n               weight=1.0):\n      self.weight = weight\n\n  def compute(\n      self, env\n  ) -> Dict[str, float]:\n      # Return per-agent\n      # reward contribution\n      # x self.weight\n      # READ-ONLY\n      ...',
     'Multiple functions combined\ninto one closure.\nWeights from YAML.'),
    (PURPLE, 'BaseAlgorithm',
     'class BaseAlgorithm:\n  def __init__(self,\n               env, config): ...\n\n  def select_actions(\n      self, obs: dict\n  ) -> dict:  # required\n\n  def train(self):\n      ...  # required\n\n  def evaluate(self,\n      episodes, max_steps):\n      ...  # inherited',
     'Only env.reset() and\nenv.step() allowed.\nNo internal reads.'),
]):
    xl = 0.4 + i * 4.32
    box(xl, 1.05, 4.12, 5.65, fill=HexColor('#F8F8F8'), stroke=col, sw=1)
    box(xl, 1.05, 4.12, 0.48, fill=col)
    wrapped_txt(title, xl+0.12, 1.09, 3.88, 0.4, size=14, color=WHITE, bold=True)
    code_block(src, xl, 1.53, 4.12, 3.95, size=11)
    box(xl, 5.5, 4.12, 1.2, fill=LGRAY, stroke=col, sw=0.5)
    wrapped_txt(note, xl+0.12, 5.58, 3.88, 1.1, size=11, color=MGRAY)

# ── Slide 15 — Black-Box Contract ────────────────────────────────────────────
new_slide('The Black-Box Algorithm Contract')
wrapped_txt('What the algorithm sees — and what it must never access.',
            0.5, 0.92, 12.33, 0.4, size=13, color=MGRAY)
box(0.4, 1.45, 5.95, 5.35, fill=LGREEN, stroke=GREEN, sw=1)
wrapped_txt('✓  Allowed', 0.65, 1.5, 5.5, 0.48, size=18, color=GREEN, bold=True)
y_ = 2.1
for a in ['env.reset(seed=n)', 'env.step(actions_dict)',
          "result['obs']", "result['reward']",
          "result['terminated']", "result['truncated']",
          "result['info']", 'env.observation_space', 'env.action_space']:
    wrapped_txt('✓  ' + a, 0.7, y_, 5.4, 0.36, size=13, color=GREEN, font='Courier')
    y_ += 0.42

box(6.9, 1.45, 5.95, 5.35, fill=LRED, stroke=RED, sw=1)
wrapped_txt('✗  Must Never Touch', 7.15, 1.5, 5.5, 0.48, size=18, color=RED, bold=True)
y_ = 2.1
for a in ['env.agents', 'env._obstacle_location', 'env._captured_agents',
          'agent._agent_location', 'agent.is_captured',
          'env.reward_fn', 'env.observation_builder']:
    wrapped_txt('✗  ' + a, 7.2, y_, 5.5, 0.36, size=13, color=RED, font='Courier')
    y_ += 0.52

box(0.4, 6.82, 12.5, 0.55, fill=NAVY)
wrapped_txt('The rule: any algorithm written here must work with any env that implements reset() + step(). '
            'That is what makes baselines reusable across all experimental conditions.',
            0.65, 6.88, 12.0, 0.44, size=12, color=WHITE)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION — RL Theory
# ══════════════════════════════════════════════════════════════════════════════
section_slide('Section 6: Connection to RL Theory',
              'From Bellman equations to running code')

# ── Slide 16 — MDP Mapping ───────────────────────────────────────────────────
new_slide('Single-Agent MDP — Every Component Maps to Code')
wrapped_txt('A Markov Decision Process is (S, A, R, P, γ). '
            'Here is exactly where each element lives in this project.',
            0.5, 0.92, 12.33, 0.4, size=13, color=MGRAY)
simple_table([
    ['MDP',              'Symbol',   'In This Codebase',                               'Source File'],
    ['State Space',      'S',        'Output of ObservationBuilder.build(env)',         'observations/*.py'],
    ['Action Space',     'A',        '5-action discrete: {Right, Up, Left, Down, Noop}','core/agent.py'],
    ['Reward Function',  'R(s,a)',   'RewardFunction.compute(env) + base_reward()',     'rewards/*.py + core/'],
    ['Transition Model', 'P(s\'|s,a)','GridWorldEnv.step() — deterministic given seed','core/gridworld.py'],
    ['Discount Factor',  'γ',   "config['experiment']['gamma']  (default 0.95)",  'configs/experiment.yaml'],
    ['Policy',           'π(a|s)','Q-table + epsilon-greedy in select_actions()', 'baselines/IQL/iql.py'],
    ['Value Function',   'Q(s,a)',   'defaultdict — states init to zero on first visit','baselines/IQL/iql.py'],
], 0.4, 1.5, 12.5, 5.2,
   col_fracs=[0.18, 0.09, 0.44, 0.29], fs=13)

# ── Slide 17 — Bellman Equation ──────────────────────────────────────────────
new_slide('The Bellman Equation — Math to Code')
wrapped_txt('The update rule that drives all tabular Q-learning in this project.',
            0.5, 0.92, 12.33, 0.4, size=13, color=MGRAY)
box(0.5, 1.4, 12.33, 1.05, fill=LGRAY, stroke=HexColor('#BBBBBB'), sw=1)
wrapped_txt('Q(s, a)  ←  Q(s, a)  +  α · [ r  +  γ · max Q(s’, a’)  −  Q(s, a) ]',
            0.7, 1.52, 12.0, 0.82, size=24, color=NAVY, bold=True, align='center')

terms = [
    (BLUE,   'α  (alpha)',    'Learning rate\nHow fast we update\nDefault: 0.1',       0.5),
    (GREEN,  'r',                  'Immediate reward\nresult["reward"]\n[agent_name]',       3.0),
    (TEAL,   'γ  (gamma)',    'Discount factor\nFuture reward weight\nDefault: 0.95',   5.5),
    (ORANGE, 'max Q(s’,·)', 'Best Q-value\nin next state\nover all actions',   8.0),
    (RED,    '− Q(s,a)',      'Current estimate\nTD error =\nbracketed term',          10.5),
]
for col, label, desc, xl in terms:
    box(xl, 2.62, 2.2, 2.1, fill=col)
    wrapped_txt(label, xl+0.1, 2.68, 2.0, 0.52, size=15, color=WHITE,
                bold=True, align='center')
    wrapped_txt(desc,  xl+0.1, 3.25, 2.0, 1.35, size=12, color=WHITE, align='center')

code_block(
    'td_error = reward + gamma * max(q_tables[agent][s_next].values()) - q_tables[agent][s][a]\n'
    'q_tables[agent][s][a] += alpha * td_error',
    0.5, 5.0, 12.33, 1.25, size=14)
wrapped_txt('q_tables is a defaultdict(lambda: defaultdict(float)) — new states auto-init to zero.',
            0.7, 6.4, 12.0, 0.42, size=12, color=MGRAY)

# ── Slide 18 — MDP to Markov Game ────────────────────────────────────────────
new_slide('From MDP to Markov Game — The MARL Jump')
wrapped_txt('Adding more agents changes the problem structure — not just its size.',
            0.5, 0.92, 12.33, 0.4, size=13, color=MGRAY)
box(0.4, 1.45, 5.95, 4.9, fill=LBLUE, stroke=BLUE, sw=1)
wrapped_txt('Single-Agent MDP', 0.62, 1.5, 5.5, 0.48, size=17, color=BLUE,
            bold=True, align='center')
y_ = 2.12
for s in ['State:   S',
          'Action:  A  (one agent)',
          'Reward:  R(s, a)',
          "Trans:   P(s' | s, a)",
          'Goal:    max Σ γᵗ Rₜ',
          '',
          'Environment is stationary:',
          'rules do not change as agent learns']:
    wrapped_txt(s, 0.65, y_, 5.5, 0.36, size=14, color=DGRAY, font='Courier')
    y_ += 0.42

box(6.9, 1.45, 5.95, 4.9, fill=HexColor('#FFF5F0'), stroke=ORANGE, sw=1)
wrapped_txt('Markov Game  (MARL)', 7.12, 1.5, 5.5, 0.48, size=17, color=ORANGE,
            bold=True, align='center')
y_ = 2.12
for s in ['State:   S  (shared by all agents)',
          'Action:  A₁ × A₂ × … × Aₙ  (joint)',
          'Reward:  Rᵢ(s, a₁,…,aₙ)  per agent',
          "Trans:   P(s' | s, a₁, …, aₙ)",
          'Goal:    each agent max Σ γᵗ Rᵢₜ',
          '',
          'Non-stationarity:',
          "from agent i's view, 'the environment'",
          'keeps changing as others learn']:
    wrapped_txt(s, 7.15, y_, 5.6, 0.36, size=13, color=DGRAY, font='Courier')
    y_ += 0.41
box(0.4, 6.45, 12.5, 0.85, fill=NAVY)
wrapped_txt('In this env: predators and prey are adversarial teams. '
            'Each team\'s reward depends on the other\'s actions. '
            'Neither side is stationary from the other\'s perspective.',
            0.65, 6.52, 12.0, 0.72, size=13, color=WHITE)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION — MARL Spectrum
# ══════════════════════════════════════════════════════════════════════════════
section_slide('Section 7: The MARL Algorithm Spectrum',
              'How the architecture enables different coordination strategies')

# ── Slide 19 — Algorithm Spectrum ────────────────────────────────════════────
new_slide('The Algorithm Spectrum — Decentralised to Centralised')
wrapped_txt('Each algorithm trades coordination ability against scalability.',
            0.5, 0.92, 12.33, 0.4, size=13, color=MGRAY)
# spectrum bar
for i, col in enumerate([BLUE, TEAL, ORANGE, RED]):
    box(0.4 + i*3.23, 1.48, 3.23, 0.42, fill=col)
wrapped_txt('Decentralised  ←────────────────────────────────────────────────  Centralised',
            0.42, 1.5, 12.46, 0.38, size=13, color=WHITE, bold=True, align='center')

algos = [
    (BLUE,   'IQL\n(Independent Q-Learning)',
     'One Q-table per agent.\nEach learns as if alone.\nIgnores teammates.',
     '✓ Scales to any n agents\n✓ Simple to implement\n✗ Non-stationarity\n✗ Coordination emergent only'),
    (TEAL,   'MixedTrainer\n(Per-Team)',
     'Predator team uses one algo,\nprey team uses another.\nDecision per team.',
     '✓ Team-level coordination\n✓ Flexible per-team choice\n✗ No cross-team model\n✗ Each team still non-stationary'),
    (ORANGE, 'CQL\n(Centralised Q-Learning)',
     'One Q-table over the joint\naction space A₁×A₂×…×Aₙ.\nFinds true joint optimum.',
     '✓ True joint optimisation\n✓ No non-stationarity\n✗ Scales as 5ⁿ actions\n✗ Impractical > 4 agents'),
    (RED,    'Nash Q-Learning\n(1v1 — Week 9)',
     'Game-theoretic equilibrium.\nEach agent best-responds\nto opponent mixed strategy.',
     '✓ Game-theoretic optimality\n✓ Handles adversarial fully\n✗ 1v1 only (needs LP solver)\n✗ General-sum is NP-hard'),
]
xl = 0.4
for col, title, desc, pros in algos:
    box(xl, 2.02, 3.1, 5.05, fill=HexColor('#F8F8F8'), stroke=col, sw=1)
    box(xl, 2.02, 3.1, 0.55, fill=col)
    wrapped_txt(title, xl+0.1, 2.05, 2.9, 0.5, size=13, color=WHITE,
                bold=True, align='center')
    wrapped_txt(desc, xl+0.12, 2.65, 2.86, 1.35, size=12, color=DGRAY)
    wrapped_txt(pros, xl+0.12, 4.1,  2.86, 1.9,  size=11, color=DGRAY)
    xl += 3.23

# ── Slide 20 — Why Architecture Enables This ─────────────────────────────────
new_slide('Why the Architecture Makes Extensibility Effortless')
wrapped_txt('Adding CQL, MixedTrainer, and Nash Q each required: one file, one registry line, one YAML. Zero core changes.',
            0.5, 0.92, 12.33, 0.4, size=13, color=MGRAY)

examples = [
    (ORANGE, 'Adding CQL to an IQL-only repo',
     'Only IQL.py in baselines/',
     '+ CQL.py in baselines/CQL/\n+ one line in algorithm_registry.py\n+ experiment_cql.yaml in configs/\n= zero changes to core, obs, rewards, or IQL'),
    (TEAL,   'Adding MixedTrainer',
     'Only IQL.py + CQL.py',
     '+ mixed.py in baselines/MIXED/\n+ one line in algorithm_registry.py\n+ experiment_mixed.yaml\n= zero changes anywhere else'),
    (PURPLE, 'Adding Nash Q-Learning (Week 9)',
     'IQL + CQL + MixedTrainer existing',
     '+ nashq.py in baselines/NASHQ/\n+ one line in algorithm_registry.py\n+ experiment_nashq.yaml\n= zero changes to prior baselines or core'),
]
y_ = 1.5
for col, title, before, after in examples:
    box(0.4, y_, 12.5, 1.65, fill=HexColor('#F9F9F9'), stroke=col, sw=1)
    wrapped_txt(title, 0.65, y_+0.07, 6.0, 0.42, size=15, color=col, bold=True)
    box(0.6, y_+0.56, 5.5, 0.98, fill=LRED, stroke=RED, sw=0.5)
    wrapped_txt('Before: ' + before, 0.75, y_+0.62, 5.2, 0.86, size=12, color=DGRAY)
    txt('→', 6.18, y_+0.92, size=22, color=col, bold=True)
    box(6.8, y_+0.56, 5.95, 0.98, fill=LGREEN, stroke=GREEN, sw=0.5)
    wrapped_txt('After: ' + after, 6.95, y_+0.62, 5.7, 0.86, size=12, color=DGRAY)
    y_ += 1.82

box(0.4, 6.82, 12.5, 0.55, fill=NAVY)
wrapped_txt('Invariant: adding an algorithm never changes the environment, plugins, or other algorithms. '
            'Scientific comparability is always preserved.',
            0.65, 6.88, 12.0, 0.44, size=12, color=WHITE)

# ── Slide 21 — Scaling Wall ───────────────────────────────────────────────────
new_slide('The Scaling Wall — When CQL Breaks Down')
wrapped_txt('Joint action spaces grow exponentially. This is why deep MARL methods exist.',
            0.5, 0.92, 12.33, 0.4, size=13, color=MGRAY)
simple_table([
    ['Setup', 'Agents (n)', 'IQL Q-tables',  'CQL Joint Actions',    'CQL Practical?'],
    ['1v1',   '2',          '2 × |S|',  '5² = 25',         'Yes'],
    ['1v2',   '3',          '3 × |S|',  '5³ = 125',         'Yes'],
    ['2v2',   '4',          '4 × |S|',  '5⁴ = 625',         'Yes (barely)'],
    ['3v3',   '6',          '6 × |S|',  '5⁶ = 15,625',      'No'],
    ['5v5',   '10',         '10 × |S|', '5¹⁰ ≈ 10M', 'Never'],
], 0.4, 1.5, 8.4, 4.3,
   col_fracs=[0.12, 0.15, 0.25, 0.28, 0.20], fs=14)

box(9.1, 1.5, 3.9, 2.35, fill=LGRAY, stroke=NAVY, sw=1)
wrapped_txt('Joint space grows as:', 9.3, 1.56, 3.5, 0.45,
            size=14, color=NAVY, bold=True, align='center')
wrapped_txt('dⁿ', 9.3, 2.05, 3.5, 0.85, size=52, color=RED,
            bold=True, align='center')
wrapped_txt('d = 5 actions  •  n = agents', 9.3, 2.98, 3.5, 0.45,
            size=13, color=DGRAY, align='center')

box(9.1, 4.0, 3.9, 1.75, fill=HexColor('#FFF5F0'), stroke=ORANGE, sw=1)
wrapped_txt('Why IQL survives:\nEach agent has its OWN\nQ-table — no joint space.\nO(n·|S|·d)  not  O(|S|·dⁿ)',
            9.3, 4.1, 3.6, 1.55, size=13, color=DGRAY)

box(0.4, 6.0, 12.5, 1.32, fill=HexColor('#FFF8F0'), stroke=ORANGE, sw=1)
wrapped_txt('This project: optimised for ≤ 4 agents on grids ≤ 10×10. '
            'For larger games, the natural next step is CTDE (Centralised Training, Decentralised Execution) '
            'methods such as QMIX or MADDPG — deep RL baselines planned for future work.',
            0.65, 6.08, 12.0, 1.18, size=13, color=DGRAY)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION — Contribution Policy
# ══════════════════════════════════════════════════════════════════════════════
section_slide('Section 8: Contributing', 'Rules, pathways, and checklist')

# ── Slide 22 — Golden Rule ───────────────────────────────────────────────────
new_slide('The Golden Rule — Immutable Core')
wrapped_txt('One rule protects the scientific integrity of every experiment run on this codebase.',
            0.5, 0.92, 12.33, 0.4, size=13, color=MGRAY)
box(0.9, 1.42, 11.53, 1.45, fill=NAVY, stroke=ORANGE, sw=2)
wrapped_txt('core/gridworld.py  and  core/agent.py  are NEVER modified by contributors.',
            1.1, 1.5, 11.1, 0.55, size=21, color=WHITE, bold=True, align='center')
wrapped_txt('All variation is expressed through config files, observation plugins, reward plugins, and algorithm plugins.',
            1.1, 2.1, 11.1, 0.65, size=16, color=AMBER, align='center')
wrapped_txt('Why?', 0.5, 3.2, 12.0, 0.45, size=17, color=NAVY, bold=True)
bullet_row([
    'Any physics change invalidates comparisons across ALL prior experiments — you lose the baseline',
    'The invariant is what makes this a fair test bed: change the rules, lose the comparison',
    'Every variation you need can be expressed through the three plugin layers',
    'If it cannot, that is a discussion for maintainers — not a unilateral edit',
    'Enforced by: code review + CONTRIBUTING.md + ADR-001',
], 0.5, 3.72, 12.33, size=16, gap=0.42)

# ── Slide 23 — Three Pathways ────────────────────────────────────────────────
new_slide('The Three Contribution Pathways')
wrapped_txt('Every contribution fits one of these patterns. Following the pathway guarantees a clean merge.',
            0.5, 0.92, 12.33, 0.4, size=13, color=MGRAY)
pathways = [
    (GREEN,  'Pathway 1: New Observation Builder', 1.5,
     '1. src/observations/my_obs.py  —  subclass ObservationBuilder, implement build(env)\n'
     '2. Add one line to  observation_registry.py\n'
     '3. Add YAML key to  configs/observations.yaml\n'
     '4. Write tests in  tests/test_observations.py'),
    (ORANGE, 'Pathway 2: New Reward Function', 1.5,
     '1. src/rewards/my_reward.py  —  subclass RewardFunction, implement compute(env)\n'
     '2. Add one line to  reward_registry.py\n'
     '3. Add YAML block to  configs/rewards.yaml\n'
     '4. Write tests in  tests/test_rewards.py'),
    (PURPLE, 'Pathway 3: New Learning Algorithm', 1.85,
     '1. src/baselines/MYALGO/myalgo.py  —  subclass BaseAlgorithm, implement select_actions() + train()\n'
     '2. Add self-registration guard at module level\n'
     '3. Import in  baselines/__init__.py\n'
     '4. Add  configs/experiment_myalgo.yaml\n'
     '5. Write tests in  tests/test_baselines_myalgo.py'),
]
y_ = 1.5
for col, title, h, steps in pathways:
    box(0.4, y_, 12.5, h, fill=HexColor('#F8F8F8'), stroke=col, sw=1)
    box(0.4, y_, 12.5, 0.46, fill=col)
    wrapped_txt(title, 0.65, y_+0.06, 12.0, 0.38, size=15, color=WHITE, bold=True)
    code_block(steps, 0.4, y_+0.47, 12.5, h-0.48, size=11)
    y_ += h + 0.12

# ══════════════════════════════════════════════════════════════════════════════
# SECTION — End-to-End Example
# ══════════════════════════════════════════════════════════════════════════════
section_slide('Section 9: End-to-End Example',
              'One command, traceable from YAML to trained policy')

# ── Slide 24 — The Scenario ──────────────────────────────────────────────────
new_slide('The Scenario — Train Predators to Capture Prey')
wrapped_txt('Goal: 2 predators learn to capture 2 prey on a 6×6 grid using IQL + distance shaping.',
            0.5, 0.92, 12.33, 0.4, size=14, color=MGRAY)
code_block('# env.yaml\ngrid_size: 6\nmax_steps: 200\ncapture_threshold: 1\nperc_num_obstacle: 15.0\nrender_mode: null\nseed: 42',
           0.4, 1.4, 3.8, 2.38, size=12)
code_block('# agents.yaml\nagents:\n  - name: pred_0\n    type: predator\n    team: predators\n  - name: pred_1\n    type: predator\n    team: predators\n  - name: prey_0\n    type: prey\n    team: prey\n  - name: prey_1\n    type: prey\n    team: prey',
           4.42, 1.4, 3.9, 4.08, size=12)
code_block('# observations.yaml\nobservation:\n  type: local_radius\n  params:\n    radius: 3\n\n# rewards.yaml\nrewards:\n  - name: base_reward\n    weight: 1.0\n  - name: predator_distance\n    weight: 0.5\n\n# experiment.yaml\nexperiment:\n  algorithm:\n    name: iql\n  episodes: 1000\n  alpha: 0.1\n  gamma: 0.95\n  epsilon: 1.0\n  epsilon_decay: 0.995\n  min_epsilon: 0.01',
           8.55, 1.4, 4.38, 5.4, size=11)
code_block('# ONE COMMAND\npython -m multi_agent_package.scripts.run_from_config --config configs/',
           0.4, 3.95, 7.85, 0.88, size=14)
wrapped_txt('The only code you write is config. The system assembles everything.',
            0.62, 5.02, 7.65, 0.42, size=13, color=MGRAY)

# ── Slide 25 — One Command ───────────────────────────────────────────────────
new_slide('One Command — Full Training Run')
wrapped_txt('What happens when you hit Enter:', 0.5, 0.92, 12.33, 0.4, size=14, color=MGRAY)
box(0.4, 1.38, 12.5, 0.72, fill=NAVY)
wrapped_txt('python -m multi_agent_package.scripts.run_from_config  --config configs/',
            0.65, 1.45, 12.0, 0.58, size=17, color=AMBER, bold=True, font='Courier')

e2e = [
    (BLUE,   '① Load',   '5 YAML files\n→ merged config'),
    (TEAL,   '② Agents', '4 Agent objects\npred_0…prey_1'),
    (GREEN,  '③ Env',    'GridWorldEnv\nsize=6, seed=42'),
    (ORANGE, '④ Obs',    'LocalRadius(r=3)\n→ env.obs_builder'),
    (PURPLE, '⑤ Reward', 'Base + Distance\n→ env.reward_fn'),
    (RED,    '⑥ Train',  'IQL.train()\n1000 episodes'),
]
bw2 = 12.5 / 6
for i, (col, step, desc) in enumerate(e2e):
    xl = 0.4 + i * bw2
    box(xl, 2.3, bw2-0.08, 0.5, fill=col)
    wrapped_txt(step, xl+0.08, 2.34, bw2-0.24, 0.42, size=13, color=WHITE,
                bold=True, align='center')
    box(xl, 2.8, bw2-0.08, 1.05, fill=LGRAY, stroke=col, sw=0.5)
    wrapped_txt(desc, xl+0.08, 2.87, bw2-0.24, 0.92, size=11, color=DGRAY,
                align='center')

box(0.4, 4.1, 12.5, 3.1, fill=HexColor('#1E1E1E'))
wrapped_txt('Console output:', 0.62, 4.16, 5.0, 0.32, size=12, color=MGRAY, bold=True)
for i, line in enumerate([
    'Episode  100/1000  |  avg_reward: -12.4  |  captures: 0.12  |  epsilon: 0.607',
    'Episode  200/1000  |  avg_reward:  -8.1  |  captures: 0.31  |  epsilon: 0.368',
    'Episode  500/1000  |  avg_reward:   2.7  |  captures: 0.64  |  epsilon: 0.082',
    'Episode 1000/1000  |  avg_reward:  14.2  |  captures: 0.88  |  epsilon: 0.010',
    'Training complete. Saved: checkpoints/iql_pred_0.pkl  (+ 3 others)',
]):
    wrapped_txt(line, 0.62, 4.6 + i*0.48, 12.0, 0.44,
                size=12, color=HexColor('#00FF41'), font='Courier')

# ── Slide 26 — Success Metrics ───────────────────────────────────────────────
new_slide('What Success Looks Like')
wrapped_txt('Same seed + same YAMLs reproduces this result exactly on any machine.',
            0.5, 0.92, 12.33, 0.4, size=13, color=MGRAY)
simple_table([
    ['Metric',                        'Trained Policy', 'Random Policy'],
    ['Capture rate',                  '88%',             '12%'],
    ['Avg episode length',            '67 steps',        '198 steps'],
    ['Avg total reward (predators)',  '+14.2',           '-8.6'],
    ['Avg total reward (prey)',       '-31.0',           '-4.1'],
    ['Unique Q-states visited',       '1,847',           '312'],
], 0.4, 1.5, 5.7, 3.9,
   col_fracs=[0.48, 0.27, 0.25], fs=14, hfill=TEAL)
code_block('python -m multi_agent_package.scripts.evaluate \\\n  --checkpoint checkpoints/ --episodes 200',
           0.4, 5.6, 5.7, 0.92, size=13)

box(6.4, 1.5, 6.7, 4.85, fill=LBLUE, stroke=BLUE, sw=1)
wrapped_txt('How to read the results', 6.62, 1.56, 6.3, 0.45,
            size=14, color=NAVY, bold=True)
insights = [
    ('Reward curve rising',     'Predators learned to move toward prey.\nDistance shaping drove this behaviour.'),
    ('Episode length falling',  'Captures happen earlier as policy improves.\nRandom agents always hit max_steps.'),
    ('States visited growing',  'Q-table coverage expands during exploration\nthen stabilises when epsilon is low.'),
    ('Prey reward drops too',   'Prey Q-tables also learn evasion —\nbut predators win this game.'),
]
y_ = 2.1
for title, body in insights:
    wrapped_txt('▶  ' + title, 6.62, y_, 6.3, 0.38, size=14, color=BLUE, bold=True)
    wrapped_txt(body, 6.62, y_+0.4, 6.3, 0.62, size=13, color=DGRAY)
    y_ += 1.12

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 27 — Status + Roadmap
# ══════════════════════════════════════════════════════════════════════════════
new_slide('Where We Are — and Where We’re Going')
wrapped_txt('Ten weeks, four students, one JOSS paper.',
            0.5, 0.92, 12.33, 0.4, size=13, color=MGRAY)

box(0.4, 1.45, 5.85, 5.35, fill=LGREEN, stroke=GREEN, sw=1)
wrapped_txt('Current State  (start of Week 1)', 0.62, 1.5, 5.5, 0.45,
            size=14, color=GREEN, bold=True)
y_ = 2.05
for s in ['Discrete gridworld (up to 10×10)',
          '5 observation builders',
          '3 reward functions',
          'IQL + CQL + MixedTrainer (tabular)',
          '189 pytest tests across 9 files',
          'Full SDD wiki (38 documents)',
          'Interactive notebooks (IQL + CQL)',
          'CI pipeline (in progress)']:
    wrapped_txt('✓  ' + s, 0.65, y_, 5.3, 0.38, size=14, color=DGRAY); y_ += 0.52

box(6.6, 1.45, 6.35, 5.35, fill=LBLUE, stroke=BLUE, sw=1)
wrapped_txt('10-Week Roadmap', 6.82, 1.5, 5.9, 0.45, size=14, color=BLUE, bold=True)
roadmap = [
    ('Week 1',    'Bug fixes  •  Gymnasium wrapper  •  CI passing'),
    ('Weeks 2–4', 'IDQN — neural Q-learning per agent'),
    ('Weeks 5–6', 'IPPO — actor-critic, policy gradient'),
    ('Weeks 7–9', 'Nash Q-Learning — 1v1 minimax'),
    ('Week 10',   'JOSS paper + final documentation'),
]
y_ = 2.05
for week, item in roadmap:
    box(6.82, y_, 5.9, 0.78, fill=LGRAY, stroke=HexColor('#BBBBBB'), sw=0.5)
    wrapped_txt(week, 6.97, y_+0.1, 1.2, 0.38, size=12, color=NAVY, bold=True, font='Courier')
    wrapped_txt(item, 8.22, y_+0.1, 4.35, 0.62, size=13, color=DGRAY)
    y_ += 0.9

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 28 — Resources
# ══════════════════════════════════════════════════════════════════════════════
slide_num[0] += 1
c.showPage()
fill_page(NAVY)
box(0, 0, 13.33, 0.32, fill=ORANGE)
box(0, 7.18, 13.33, 0.32, fill=ORANGE)
wrapped_txt('Resources & Next Steps', 0.5, 0.55, 12.33, 0.85,
            size=32, color=WHITE, bold=True, align='center')
hline(1.0, 1.55, 11.33, ORANGE, lw=1.5)

cards = [
    (BLUE,   'Start Here', [
        'wiki/students/index.md  —  curated student documentation',
        'notebooks/01_iql_demo.ipynb  —  interactive IQL walk-through',
        'guides/quickstart.md  —  install + first run in 5 minutes',
    ]),
    (TEAL,   'Key References', [
        'Sutton & Barto (2018)  —  Chapters 6, 13',
        'Hu & Wellman (2003)  —  Nash Q-Learning',
        'Littman (2001)  —  Friend-or-Foe Q-Learning',
    ]),
    (ORANGE, 'This Week', [
        'wiki/weekly/week-01.md  —  task list with owners',
        'wiki/reviews/audit-2026-06-07.md  —  known bugs',
        'CONTRIBUTING.md  —  pull request checklist',
    ]),
]
xl = 0.5
for col, title, items in cards:
    box(xl, 1.85, 4.0, 5.2, fill=HexColor('#0D2238'), stroke=col, sw=1)
    box(xl, 1.85, 4.0, 0.5, fill=col)
    wrapped_txt(title, xl+0.15, 1.88, 3.7, 0.42, size=15, color=WHITE, bold=True)
    y_ = 2.52
    for item in items:
        wrapped_txt('•  ' + item, xl+0.15, y_, 3.7, 0.65,
                    size=12, color=HexColor('#CCE0EE'))
        y_ += 0.78
    xl += 4.4

wrapped_txt('Build understanding from the ground up.',
            0.5, 7.1, 12.33, 0.35, size=15, color=AMBER, align='center')

# ══════════════════════════════════════════════════════════════════════════════
# Page numbers on content slides
# ══════════════════════════════════════════════════════════════════════════════
c.save()
print(f'Saved {slide_num[0]} slides -> {OUT}')
