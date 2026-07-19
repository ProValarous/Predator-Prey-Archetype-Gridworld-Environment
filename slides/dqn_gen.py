#!/usr/bin/env python3
"""DQN Lecture Slides — compact, clean style.  Run to produce dqn_lecture.pdf"""

from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor, white, black
import math, os

# ── fonts (Arial for Unicode / Greek / math) ──────────────────────────────────
FONT_DIR = r"C:\Windows\Fonts"
_fonts = {
    "R":  ("Arial",       os.path.join(FONT_DIR, "arial.ttf")),
    "B":  ("Arial-Bold",  os.path.join(FONT_DIR, "arialbd.ttf")),
    "I":  ("Arial-Italic",os.path.join(FONT_DIR, "ariali.ttf")),
}
for alias, (name, path) in _fonts.items():
    try:
        pdfmetrics.registerFont(TTFont(name, path))
    except Exception:
        pass  # fall back to Helvetica

def FR(sz=12):  return "Arial"       if "Arial" in pdfmetrics.getRegisteredFontNames() else "Helvetica"
def FB(sz=12):  return "Arial-Bold"  if "Arial-Bold" in pdfmetrics.getRegisteredFontNames() else "Helvetica-Bold"
def FI(sz=12):  return "Arial-Italic"if "Arial-Italic" in pdfmetrics.getRegisteredFontNames() else "Helvetica-Oblique"

# ── slide dimensions (16 : 9) ─────────────────────────────────────────────────
W, H   = 960, 540
TH     = 60     # title-bar height
FH     = 20     # footer height
ML     = 40     # left margin
TOTAL  = 30     # total slides (update if you add/remove)

# ── palette ───────────────────────────────────────────────────────────────────
NAVY   = HexColor("#0D2137")
BLUE   = HexColor("#1565C0")
ACT    = HexColor("#1976D2")
LBLUE  = HexColor("#DEEFFE")
ORANGE = HexColor("#E65100")
LORG   = HexColor("#FFF3E0")
GREEN  = HexColor("#2E7D32")
LGRN   = HexColor("#E8F5E9")
RED    = HexColor("#C62828")
LRED   = HexColor("#FFEBEE")
PURPLE = HexColor("#6A1B9A")
LPUR   = HexColor("#F3E5F5")
TEAL   = HexColor("#00695C")
LTEAL  = HexColor("#E0F2F1")
GRAY   = HexColor("#546E7A")
MGRY   = HexColor("#90A4AE")
LGRY   = HexColor("#ECEFF1")
BODY   = HexColor("#1C1C2E")
BG     = HexColor("#FAFAFA")
GOLD   = HexColor("#F57F17")
LYEL   = HexColor("#FFFDE7")


# ─────────────────────────────────────────────────────────────────────────────
class D:
    """Thin slide-deck helper wrapping ReportLab canvas."""

    def __init__(self, path):
        self.cv = canvas.Canvas(path, pagesize=(W, H))
        self.n  = 0

    # ── page / header ─────────────────────────────────────────────────────────
    def page(self, bg=BG):
        if self.n:
            self.cv.showPage()
        self.n += 1
        self.cv.setFillColor(bg)
        self.cv.rect(0, 0, W, H, fill=1, stroke=0)
        return self

    def hdr(self, title, sec=""):
        cv = self.cv
        cv.setFillColor(NAVY);  cv.rect(0, H - TH, W, TH, fill=1, stroke=0)
        cv.setFillColor(ACT);   cv.rect(0, H - TH - 3, W, 3, fill=1, stroke=0)
        cv.setFillColor(white); cv.setFont(FB(), 19)
        cv.drawString(ML, H - TH + 17, title)
        if sec:
            cv.setFillColor(MGRY); cv.setFont(FR(), 8)
            cv.drawRightString(W - 16, H - 12, sec.upper())
        cv.setFillColor(MGRY); cv.setFont(FR(), 8)
        cv.drawRightString(W - 12, 5, f"{self.n} / {TOTAL}")
        cv.setStrokeColor(LGRY); cv.setLineWidth(0.4)
        cv.line(0, FH, W, FH)
        return self

    # ── primitives ────────────────────────────────────────────────────────────
    def tx(self, x, y, s, bold=False, italic=False, sz=12, cl=None, a="l"):
        cv = self.cv
        cv.setFillColor(cl or BODY)
        font = FB() if bold else (FI() if italic else FR())
        cv.setFont(font, sz)
        draw = {"c": cv.drawCentredString, "r": cv.drawRightString}.get(a, cv.drawString)
        for i, line in enumerate(str(s).split("\n")):
            draw(x, y - i * (sz + 4), line)

    def bul(self, x, y, txt, lv=0, sz=12, cl=None, bold=False):
        mk  = ["▶", "–", "·"][min(lv, 2)]
        mkc = [ACT, BLUE, GRAY][min(lv, 2)]
        ind = lv * 22
        self.tx(x + ind,      y, mk,  sz=sz - 1, cl=mkc)
        self.tx(x + ind + 16, y, txt, bold=bold, sz=sz, cl=cl or BODY)
        return y - sz - 9

    def rb(self, x, y, w, h, fill, sk=None, r=5, lw=1):
        cv = self.cv
        cv.setFillColor(fill)
        if sk:
            cv.setStrokeColor(sk); cv.setLineWidth(lw)
        cv.roundRect(x, y, w, h, r, fill=1, stroke=1 if sk else 0)

    def eqn(self, x, y, w, h, lines, cap="", bg=LBLUE):
        """Equation box: lines is a list of formula strings (or a single str)."""
        if isinstance(lines, str):
            lines = [l for l in lines.split("\n") if l.strip()]
        self.rb(x, y, w, h, bg, r=5)
        cv = self.cv
        cv.setStrokeColor(ACT); cv.setLineWidth(2.5)
        cv.line(x + 5, y + 5, x + 5, y + h - 5)
        n = len(lines)
        step = min(18, (h - 20) / max(n, 1))
        top = y + h / 2 + (n - 1) * step / 2
        for i, ln in enumerate(lines):
            self.tx(x + w / 2, top - i * step, ln, bold=True, sz=12, cl=BLUE, a="c")
        if cap:
            self.tx(x + w / 2, y + 5, cap, italic=True, sz=8, cl=GRAY, a="c")

    def arr(self, x1, y1, x2, y2, cl=None, lw=1.5, dash=None):
        cv = self.cv
        cv.setStrokeColor(cl or ACT); cv.setLineWidth(lw)
        if dash:
            cv.setDash(dash)
        cv.line(x1, y1, x2, y2)
        cv.setDash()
        a = math.atan2(y2 - y1, x2 - x1); s = 8
        cv.setFillColor(cl or ACT)
        p = cv.beginPath()
        p.moveTo(x2, y2)
        p.lineTo(x2 - s * math.cos(a - 0.4), y2 - s * math.sin(a - 0.4))
        p.lineTo(x2 - s * math.cos(a + 0.4), y2 - s * math.sin(a + 0.4))
        p.close()
        cv.drawPath(p, fill=1, stroke=0)

    def badge(self, x, y, w, h, title, body_lines, th=BLUE, bh=LBLUE):
        self.rb(x, y + h - 26, w, 26, th, r=4)
        self.tx(x + 8, y + h - 17, title, bold=True, sz=10, cl=white)
        self.rb(x, y, w, h - 26, bh, r=4)
        cy = y + h - 44
        for ln in body_lines:
            self.tx(x + 8, cy, ln, sz=9); cy -= 13

    def save(self):
        self.cv.save()
        print(f"  Written {self.n} slides.")


# ─────────────────────────────────────────────────────────────────────────────
def build(path):
    d  = D(path)
    cv = d.cv
    CT = H - TH - 14    # content region top-y
    CB = FH + 10         # content region bottom-y

    # ══════════════════════════════════════════════════════════════
    #  1. TITLE
    # ══════════════════════════════════════════════════════════════
    d.page(NAVY)
    cv.setFillColor(HexColor("#1E3A5F"))
    cv.roundRect(0, 0, 5, H, 0, fill=1, stroke=0)
    cv.setFillColor(HexColor("#FFFFFF09"))
    cv.roundRect(640, 60, 320, 320, 160, fill=1, stroke=0)

    d.tx(W // 2, 360, "Deep Q-Networks (DQN)", bold=True, sz=42, cl=white, a="c")
    d.tx(W // 2, 308, "A Lecture on Deep Reinforcement Learning",
         italic=True, sz=16, cl=MGRY, a="c")
    cv.setStrokeColor(ACT); cv.setLineWidth(1)
    cv.line(180, 294, 780, 294)
    d.tx(W // 2, 268,
         "Why DL? · Q-Learning · Architecture · Components · Flow · Variants",
         sz=12, cl=HexColor("#78909C"), a="c")
    d.tx(W // 2, 70, "Predator–Prey Archetype Gridworld  ·  2026",
         sz=10, cl=HexColor("#455A64"), a="c")
    d.tx(12, 5, "1 / 30", sz=8, cl=MGRY)

    # ══════════════════════════════════════════════════════════════
    #  2. OUTLINE
    # ══════════════════════════════════════════════════════════════
    d.page(); d.hdr("Lecture Outline")
    secs = [
        ("1",  "Why Deep Learning?",        "Tabular limits & function approximation"),
        ("2",  "RL Fundamentals",           "MDP, returns, value functions, Bellman"),
        ("3",  "TD Learning & Q-Learning",  "TD updates, off-policy control"),
        ("4",  "The DQN Insight",           "Scaling RL to high-dim spaces"),
        ("5",  "DQN Architecture",          "CNN + FC network design"),
        ("6",  "DQN Components",            "Replay, target net, ε-greedy, loss"),
        ("7",  "End-to-End Flow",           "Training algorithm & pipeline"),
        ("8",  "Example: CartPole",         "Worked example"),
        ("9",  "DQN Variants",              "DDQN, Dueling, PER, Noisy, Rainbow"),
        ("10", "Key Takeaways",             "Summary & further reading"),
    ]
    BW, BH, GAP = 400, 34, 6
    for i, (num, name, detail) in enumerate(secs):
        col = i // 5; row = i % 5
        x = ML + col * (BW + 55)
        y = CT - row * (BH + GAP)
        bg = LBLUE if col == 0 else LGRN
        ac = ACT  if col == 0 else GREEN
        d.rb(x, y - BH + 4, BW, BH, bg, r=4)
        d.tx(x + 8,  y - BH + 20, f"{num}.", bold=True, sz=11, cl=ac)
        d.tx(x + 30, y - BH + 20, name, bold=True, sz=11, cl=BODY)
        d.tx(x + 30, y - BH + 7,  detail, sz=9, cl=GRAY)

    # ══════════════════════════════════════════════════════════════
    #  SECTION 1 — WHY DEEP LEARNING?
    # ══════════════════════════════════════════════════════════════
    S1 = "Why Deep Learning?"

    # ── 3. Limits of Tabular RL ───────────────────────────────────
    d.page(); d.hdr("The Limits of Tabular RL", S1)
    y = CT
    d.tx(ML, y, "A Q-table stores one value per (state, action) pair.", sz=13)
    y -= 26
    y = d.bul(ML, y, "Works for small discrete worlds: Gridworld, FrozenLake, …", sz=12)
    y = d.bul(ML, y, "Exact lookup — convergence guaranteed", sz=12)
    y -= 16

    # mini Q-table
    cols_lbl = ["←", "→", "↑", "↓"]
    rows_lbl  = ["s₀", "s₁", "s₂", "s₃", "⋮"]
    CW, RH = 68, 28; gx = ML + 30; gy = y - 14
    d.rb(gx, gy - len(rows_lbl) * RH, (len(cols_lbl) + 1) * CW,
         (len(rows_lbl) + 1) * RH, LGRY, r=4)
    for j, lbl in enumerate(cols_lbl):
        d.rb(gx + (j + 1) * CW, gy - RH, CW, RH, NAVY, r=2)
        d.tx(gx + (j + 1) * CW + CW // 2, gy - RH + 9, lbl, bold=True, sz=11, cl=white, a="c")
    d.tx(gx + CW // 2, gy - RH + 9, "Q(s,a)", bold=True, sz=8, cl=GRAY, a="c")
    vals = [["0.12","0.85","0.23","0.07"],
            ["0.44","0.31","0.62","0.55"],
            ["0.08","0.19","0.91","0.03"],
            ["0.73","0.65","0.44","0.88"],
            ["…","…","…","…"]]
    for i, row in enumerate(rows_lbl):
        ry2 = gy - (i + 2) * RH
        d.rb(gx, ry2, CW, RH, LGRY, r=2)
        d.tx(gx + CW // 2, ry2 + 9, row, bold=True, sz=10, cl=BODY, a="c")
        for j, v in enumerate(vals[i]):
            hi = v != "…" and float(v) > 0.7 if v != "…" else False
            d.rb(gx + (j + 1) * CW, ry2, CW, RH, LGRN if hi else BG, r=2)
            d.tx(gx + (j + 1) * CW + CW // 2, ry2 + 9, v,
                 sz=9, cl=GREEN if hi else GRAY, a="c")

    # problem callout (right column)
    px = gx + (len(cols_lbl) + 1) * CW + 50
    py = gy - len(rows_lbl) * RH - 4
    ph = len(rows_lbl) * RH + RH + 8
    d.rb(px, py, W - px - ML, ph, LRED, sk=RED, r=6)
    d.tx(px + 12, py + ph - 22, "⚠  The Problem", bold=True, sz=12, cl=RED)
    for i, ln in enumerate(["Most real problems have",
                             "enormous or continuous",
                             "state spaces.",
                             "",
                             "Atari:  ≈ 10⁷⁰⁰⁰⁰ states",
                             "A Q-table cannot scale."]):
        d.tx(px + 12, py + ph - 46 - i * 18, ln, sz=10, cl=BODY if ln else BODY)

    # ── 4. Curse of Dimensionality ────────────────────────────────
    d.page(); d.hdr("The Curse of Dimensionality", S1)
    y = CT
    d.tx(ML, y, "State space grows exponentially with problem complexity.", sz=13)
    y -= 30

    items = [("Simple Grid", "4 × 4 = 16 states",   GREEN,  LGRN),
             ("Atari Pixels", "≈ 10⁷⁰,⁰⁰⁰ states", RED,    LRED),
             ("Continuous",  "∞ states",             PURPLE, LPUR)]
    bw4 = (W - 2 * ML - 40) // 3
    for i, (name, val, tc, bc) in enumerate(items):
        bx = ML + i * (bw4 + 20); by = y - 150
        d.rb(bx, by, bw4, 150, bc, sk=tc, r=8)
        d.rb(bx, by + 122, bw4, 28, tc, r=8)
        d.tx(bx + bw4 // 2, by + 133, name, bold=True, sz=12, cl=white, a="c")
        d.tx(bx + bw4 // 2, by + 80,  val,  bold=True, sz=14, cl=tc,    a="c")

    y = y - 170
    d.rb(ML, y - 38, W - 2 * ML, 44, LYEL, sk=GOLD, r=5)
    d.tx(ML + 14, y - 12, "💡  Key insight:", bold=True, sz=12, cl=GOLD)
    d.tx(ML + 140, y - 12,
         "We need to generalise across states — not memorise each one.", sz=12)

    # ── 5. Function Approximation ─────────────────────────────────
    d.page(); d.hdr("Deep Learning as Function Approximator", S1)
    y = CT
    d.tx(ML, y, "Replace the Q-table with a neural network that generalises.", sz=13)
    y -= 26

    hw = (W - 2 * ML - 30) // 2
    # Q-table
    d.rb(ML, y - 155, hw, 155, LRED, sk=RED, r=6)
    d.tx(ML + hw // 2, y - 18, "Q-Table", bold=True, sz=13, cl=RED, a="c")
    for i, ln in enumerate(["× One row per state",
                             "× No generalisation",
                             "× Breaks at scale"]):
        d.tx(ML + 20, y - 50 - i * 30, ln, sz=12, cl=RED)

    # Q-Network
    rx = ML + hw + 30
    d.rb(rx, y - 155, hw, 155, LGRN, sk=GREEN, r=6)
    d.tx(rx + hw // 2, y - 18, "Q-Network  Q(s,a; θ)", bold=True, sz=13, cl=GREEN, a="c")
    for i, ln in enumerate(["✓ Generalises across states",
                             "✓ Handles images/sensors",
                             "✓ Scalable"]):
        d.tx(rx + 20, y - 50 - i * 30, ln, sz=12, cl=GREEN)

    d.arr(ML + hw + 2, y - 78, rx - 2, y - 78, cl=ACT, lw=2)
    d.tx(ML + hw + 15, y - 72, "→", bold=True, sz=14, cl=ACT, a="c")

    y -= 172
    d.eqn(ML, y - 44, W - 2 * ML, 50,
          "Q*(s, a)  ≈  Q(s, a ; θ)     where θ are the network weights",
          "The network takes state s and outputs Q-values for every action a")

    # ══════════════════════════════════════════════════════════════
    #  SECTION 2 — RL FUNDAMENTALS
    # ══════════════════════════════════════════════════════════════
    S2 = "RL Fundamentals"

    # ── 6. MDP / RL Framework ────────────────────────────────────
    d.page(); d.hdr("The RL Framework: Markov Decision Process", S2)

    # Agent ↔ Environment diagram
    ax = 170; ay = 230; aw = 190; ah = 90
    ex = 590; ey = 230; ew = 190; eh = 90
    d.rb(ax, ay, aw, ah, LBLUE, sk=BLUE, r=8)
    d.tx(ax + aw // 2, ay + ah // 2 + 4, "Agent", bold=True, sz=16, cl=BLUE, a="c")
    d.tx(ax + aw // 2, ay + 14, "policy π(a|s)", sz=10, cl=GRAY, a="c")

    d.rb(ex, ey, ew, eh, LGRN, sk=GREEN, r=8)
    d.tx(ex + ew // 2, ey + eh // 2 + 4, "Environment", bold=True, sz=16, cl=GREEN, a="c")
    d.tx(ex + ew // 2, ey + 14, "P(s'|s,a),  R(s,a)", sz=10, cl=GRAY, a="c")

    mid_y = ay + ah // 2
    d.arr(ax + aw, mid_y + 18, ex, mid_y + 18, cl=ORANGE, lw=2)
    d.tx(W // 2, mid_y + 30, "action  aₜ", bold=True, sz=11, cl=ORANGE, a="c")
    d.arr(ex, mid_y - 18, ax + aw, mid_y - 18, cl=PURPLE, lw=2)
    d.tx(W // 2, mid_y - 30, "next state sₜ₊₁,  reward rₜ", bold=True, sz=11, cl=PURPLE, a="c")

    d.tx(ML, CB + 20, "MDP tuple:", bold=True, sz=11)
    d.tx(ML + 95, CB + 20,
         "(S, A, P, R, γ)  —  States, Actions, Transitions, Rewards, Discount",
         sz=11)

    # ── 7. Return & Discount ─────────────────────────────────────
    d.page(); d.hdr("Return & the Discount Factor γ", S2)
    y = CT
    d.tx(ML, y, "The agent maximises the expected discounted cumulative reward:", sz=13)
    y -= 30
    d.eqn(ML, y - 52, W - 2 * ML, 58,
          "Gₜ  =  rₜ + γ rₜ₊₁ + γ² rₜ₊₂ + …  =  Σₖ₌₀^∞  γᵏ rₜ₊ₖ",
          "G_t = return from timestep t    γ ∈ [0,1) = discount factor")
    y -= 72

    items7 = [("γ = 0",  "Myopic — only\nnext reward matters",    ORANGE, LORG),
              ("γ → 1",  "Far-sighted — all\nfuture rewards equal", GREEN,  LGRN),
              ("γ = 0.99","Typical choice\n(balances both)",       BLUE,   LBLUE)]
    bw7 = (W - 2 * ML - 40) // 3
    for i, (lbl, desc, tc, bc) in enumerate(items7):
        bx = ML + i * (bw7 + 20); by = y - 130
        d.rb(bx, by, bw7, 130, bc, sk=tc, r=6)
        d.rb(bx, by + 104, bw7, 26, tc, r=6)
        d.tx(bx + bw7 // 2, by + 114, lbl, bold=True, sz=13, cl=white, a="c")
        for j, ln in enumerate(desc.split("\n")):
            d.tx(bx + bw7 // 2, by + 72 - j * 20, ln, sz=11, cl=tc, a="c")

    # ── 8. Value Functions ────────────────────────────────────────
    d.page(); d.hdr("Value Functions: V(s) and Q(s,a)", S2)
    y = CT
    d.tx(ML, y, "Two key functions describe how good a state or action is:", sz=13)
    y -= 26

    hw8 = (W - 2 * ML - 30) // 2
    # V(s)
    d.rb(ML, y - 168, hw8, 168, LBLUE, sk=BLUE, r=8)
    d.tx(ML + hw8 // 2, y - 20, "State-Value  V(s)", bold=True, sz=13, cl=BLUE, a="c")
    d.eqn(ML + 10, y - 100, hw8 - 20, 48,
          "Vᵖ(s) = E_π [Gₜ | sₜ = s]",
          "Expected return from state s under policy π", LBLUE)
    d.tx(ML + 14, y - 118, '"How good is it to be in state s?"',
         italic=True, sz=10, cl=GRAY)
    d.tx(ML + 14, y - 152, "Used in policy evaluation", sz=10)

    # Q(s,a)
    rx8 = ML + hw8 + 30
    d.rb(rx8, y - 168, hw8, 168, LGRN, sk=GREEN, r=8)
    d.tx(rx8 + hw8 // 2, y - 20, "Action-Value  Q(s,a)", bold=True, sz=13, cl=GREEN, a="c")
    d.eqn(rx8 + 10, y - 100, hw8 - 20, 48,
          "Qᵖ(s,a) = E_π [Gₜ | sₜ=s, aₜ=a]",
          "Expected return taking action a in state s, then following π", LGRN)
    d.tx(rx8 + 14, y - 118, '"How good is action a in state s?"',
         italic=True, sz=10, cl=GRAY)
    d.tx(rx8 + 14, y - 152, "DQN approximates Q*(s,a) directly!", bold=True, sz=10, cl=GREEN)

    y -= 184
    d.rb(ML, y - 34, W - 2 * ML, 40, LYEL, sk=GOLD, r=5)
    d.tx(ML + 14, y - 12, "Optimal policy:  π*(s) = argmaxₐ Q*(s,a)  — just pick the highest Q-value",
         sz=12)

    # ── 9. Bellman Equation ───────────────────────────────────────
    d.page(); d.hdr("The Bellman Optimality Equation", S2)
    y = CT
    d.tx(ML, y, "The optimal Q-function satisfies a recursive self-consistency equation:", sz=13)
    y -= 30
    d.eqn(ML, y - 56, W - 2 * ML, 62,
          "Q*(s, a) = E [ r + γ · maxₐ' Q*(s', a')  |  s, a ]",
          "Value of (s,a) = immediate reward + discounted best achievable value from next state")
    y -= 74

    d.tx(ML, y, "Intuition:", bold=True, sz=13)
    y -= 20
    y = d.bul(ML, y, "Q*(s,a) is defined in terms of itself — a recursive equation", sz=12)
    y = d.bul(ML, y, "Solved iteratively by repeated application (Q-learning, Value Iteration)", sz=12)
    y = d.bul(ML, y, "DQN trains a neural network to satisfy this equation", sz=12)
    y -= 16
    d.rb(ML, y - 40, W - 2 * ML, 46, LORG, sk=ORANGE, r=5)
    d.tx(ML + 14, y - 14, "→  Q-Learning solves this iteratively with tabular updates.", sz=12)
    d.tx(ML + 14, y - 32, "→  DQN approximates Q* with a neural network trained by gradient descent.", sz=12)

    # ══════════════════════════════════════════════════════════════
    #  SECTION 3 — TD & Q-LEARNING
    # ══════════════════════════════════════════════════════════════
    S3 = "TD Learning & Q-Learning"

    # ── 10. TD Learning ──────────────────────────────────────────
    d.page(); d.hdr("Temporal Difference (TD) Learning", S3)
    y = CT
    d.tx(ML, y, "TD learning updates value estimates online — no episode end needed.", sz=13)
    y -= 30
    d.eqn(ML, y - 54, W - 2 * ML, 60,
          ["V(sₜ) ← V(sₜ) + α · δₜ",
           "where  δₜ = rₜ + γ V(sₜ₊₁) − V(sₜ)   (TD error)"],
          "α = step size / learning rate")
    y -= 72

    hw10 = (W - 2 * ML - 20) // 2
    d.rb(ML, y - 130, hw10, 130, LORG, sk=ORANGE, r=6)
    d.tx(ML + hw10 // 2, y - 20, "Monte Carlo", bold=True, sz=12, cl=ORANGE, a="c")
    for i, ln in enumerate(["Wait until episode ends",
                             "Use full return Gₜ",
                             "High variance, unbiased"]):
        d.tx(ML + 14, y - 50 - i * 26, ln, sz=11)

    rx10 = ML + hw10 + 20
    d.rb(rx10, y - 130, hw10, 130, LGRN, sk=GREEN, r=6)
    d.tx(rx10 + hw10 // 2, y - 20, "TD Learning", bold=True, sz=12, cl=GREEN, a="c")
    for i, (ln, bold) in enumerate([
        ("Update after each step", False),
        ("Bootstrap from V(sₜ₊₁)", False),
        ("Lower variance ✓", True)
    ]):
        d.tx(rx10 + 14, y - 50 - i * 26, ln, bold=bold, sz=11, cl=GREEN if bold else BODY)

    # ── 11. Q-Learning ────────────────────────────────────────────
    d.page(); d.hdr("Q-Learning: Off-Policy TD Control", S3)
    y = CT
    d.tx(ML, y, "Q-learning applies TD directly to learn Q*(s,a) without a model.", sz=13)
    y -= 30
    d.eqn(ML, y - 58, W - 2 * ML, 64,
          "Q(sₜ, aₜ) ← Q(sₜ, aₜ)  +  α [ rₜ  +  γ maxₐ Q(sₜ₊₁, a)  −  Q(sₜ, aₜ) ]",
          "α = learning rate     maxₐ = greedy selector     γ = discount")
    y -= 76

    d.tx(ML, y, "Key properties:", bold=True, sz=12)
    y -= 20
    y = d.bul(ML, y, "Off-policy: learns Q* regardless of which policy collects data", sz=12)
    y = d.bul(ML, y, "Model-free: doesn't need to know P(s'|s,a) or R(s,a)", sz=12)
    y = d.bul(ML, y, "Provably converges to Q* given sufficient exploration + decaying α", sz=12)
    y -= 16
    d.rb(ML, y - 36, W - 2 * ML, 42, LYEL, sk=GOLD, r=5)
    d.tx(ML + 14, y - 12, "🔑  Q-Learning is the direct predecessor of DQN.", bold=True, sz=12)
    d.tx(ML + 14, y - 28, "    DQN replaces the Q-table with a deep neural network.", sz=11, cl=GRAY)

    # ── 12. Why Q-Learning fails at scale ─────────────────────────
    d.page(); d.hdr("Why Q-Learning Fails at Scale", S3)
    y = CT
    d.tx(ML, y, "Naively adding a neural network to Q-learning breaks training.", sz=13)
    y -= 24

    problems = [
        ("Problem 1 · Correlated data",
         "Consecutive (s,a,r,s') are highly correlated.\n"
         "Neural nets need i.i.d. samples → divergence.",
         LRED, RED),
        ("Problem 2 · Moving targets",
         "Target r+γ maxQ(s';θ) shifts every update.\n"
         "Chasing a moving target → instability.",
         LORG, ORANGE),
        ("Problem 3 · Catastrophic forgetting",
         "Updates on new data overwrite old knowledge.\n"
         "Net forgets what it learned earlier.",
         LPUR, PURPLE),
    ]
    ph = 88
    for i, (title, body, bc, tc) in enumerate(problems):
        by = y - 20 - i * (ph + 10)
        d.rb(ML, by - ph, W - 2 * ML, ph, bc, sk=tc, r=6)
        d.rb(ML, by - ph, 160, ph, tc, r=6)
        d.tx(ML + 8, by - 26, title, bold=True, sz=10, cl=white)
        for j, ln in enumerate(body.split("\n")):
            d.tx(ML + 174, by - 30 - j * 20, ln, sz=11)

    y -= 30 + 3 * (ph + 10) + 6
    d.rb(ML, y - 32, W - 2 * ML, 38, LGRN, sk=GREEN, r=5)
    d.tx(ML + 14, y - 10, "✓  DQN solves Problems 1 & 2 with Experience Replay + Target Network.", sz=12, cl=GREEN)

    # ══════════════════════════════════════════════════════════════
    #  SECTION 4 — THE DQN INSIGHT
    # ══════════════════════════════════════════════════════════════
    S4 = "The DQN Insight"

    # ── 13. DQN: The Big Idea ─────────────────────────────────────
    d.page(); d.hdr("DQN: The Big Idea", S4)
    y = CT
    d.tx(ML, y, "DQN = Q-Learning  +  Deep Neural Network  +  two stabilising tricks.", sz=13)
    y -= 28

    pillars = [
        ("Q-Learning",  "Off-policy TD\ncontrol",          BLUE,   LBLUE),
        ("Deep NN",     "Q-value function\napproximator",   GREEN,  LGRN),
        ("2 Fixes",     "Replay Buffer\n+ Target Network",  ORANGE, LORG),
    ]
    bw13 = (W - 2 * ML - 40) // 3
    for i, (name, desc, tc, bc) in enumerate(pillars):
        bx = ML + i * (bw13 + 20); by = y - 140
        d.rb(bx, by, bw13, 140, bc, sk=tc, r=8)
        d.rb(bx, by + 112, bw13, 28, tc, r=8)
        d.tx(bx + bw13 // 2, by + 122, name, bold=True, sz=13, cl=white, a="c")
        for j, ln in enumerate(desc.split("\n")):
            d.tx(bx + bw13 // 2, by + 78 - j * 22, ln, sz=11, cl=tc, a="c")
        if i < 2:
            px = bx + bw13 + 10
            d.tx(px, by + 70, "+", bold=True, sz=18, cl=GRAY, a="c")

    y -= 158
    d.eqn(ML, y - 46, W - 2 * ML, 52,
          "Action = argmaxₐ  Q(s, a ; θ)    trained with ε-greedy exploration",
          "θ = neural network weights learned via gradient descent on the Bellman error")

    # ── 14. The Atari Breakthrough ────────────────────────────────
    d.page(); d.hdr("The Atari Breakthrough  (Mnih et al., Nature 2015)", S4)
    y = CT
    d.tx(ML, y, "DeepMind trained one DQN agent on 49 Atari 2600 games.", sz=13)
    y -= 24

    rows14 = [
        ("Input",      "Raw 84×84 greyscale pixels — no hand-crafted features"),
        ("Output",     "Q-value per joystick action"),
        ("Training",   "Same hyperparameters for every game"),
        ("Result",     "Superhuman on 29 / 49 games"),
        ("Published",  "Nature, February 2015 — landmark paper for deep RL"),
    ]
    for label, desc in rows14:
        d.rb(ML, y - 32, W - 2 * ML, 30, LGRY, r=4)
        d.tx(ML + 10, y - 18, label + ":", bold=True, sz=11, cl=BLUE)
        d.tx(ML + 110, y - 18, desc, sz=11)
        y -= 36

    y -= 8
    d.rb(ML, y - 44, W - 2 * ML, 50, LGRN, sk=GREEN, r=6)
    d.tx(ML + 14, y - 16, "🏆  One algorithm, raw pixels, superhuman performance on diverse games.",
         sz=12, cl=GREEN)
    d.tx(ML + 14, y - 32, "    Demonstrated that deep RL can generalise end-to-end across tasks.",
         sz=11, cl=GRAY)

    # ══════════════════════════════════════════════════════════════
    #  SECTION 5 — DQN ARCHITECTURE
    # ══════════════════════════════════════════════════════════════
    S5 = "DQN Architecture"

    # ── 15. Architecture Overview ─────────────────────────────────
    d.page(); d.hdr("DQN Network Architecture", S5)

    ym = H // 2 - 10     # vertical centre of diagram

    # ── blocks ──
    def block(bx, by, bw, bh, label, sub, fill, stroke):
        d.rb(bx, by, bw, bh, fill, sk=stroke, r=5)
        d.tx(bx + bw // 2, by + bh // 2 + 6, label, bold=True, sz=11, cl=stroke, a="c")
        d.tx(bx + bw // 2, by + bh // 2 - 10, sub,  sz=8,       cl=GRAY,   a="c")

    IN_X = ML + 10; IN_W = 90; IN_H = 110
    block(IN_X, ym - IN_H // 2, IN_W, IN_H, "Input", "84×84×4", LPUR, PURPLE)
    d.arr(IN_X + IN_W, ym, IN_X + IN_W + 24, ym, cl=GRAY)

    conv_specs = [("Conv1", "32 filt\n8×8"), ("Conv2", "64 filt\n4×4"), ("Conv3", "64 filt\n3×3")]
    CH = [110, 90, 70]; CX_s = IN_X + IN_W + 24; CW = 84
    for i, (name, sub) in enumerate(conv_specs):
        bx = CX_s + i * (CW + 18)
        block(bx, ym - CH[i] // 2, CW, CH[i], name, sub, LBLUE, BLUE)
        if i < 2:
            d.arr(bx + CW, ym, bx + CW + 18, ym, cl=GRAY)
    cx_after_conv = CX_s + 3 * (CW + 18) - 18 + CW
    d.arr(cx_after_conv, ym, cx_after_conv + 22, ym, cl=GRAY)

    FC_X = cx_after_conv + 22; FC_W = 86; FC_H = 90
    block(FC_X, ym - FC_H // 2, FC_W, FC_H, "FC 512", "ReLU", LGRN, GREEN)
    d.arr(FC_X + FC_W, ym, FC_X + FC_W + 22, ym, cl=GRAY)

    OUT_X = FC_X + FC_W + 22; OUT_W = 110; OUT_H = 110
    d.rb(OUT_X, ym - OUT_H // 2, OUT_W, OUT_H, LORG, sk=ORANGE, r=5)
    d.tx(OUT_X + OUT_W // 2, ym + OUT_H // 2 - 16, "Output", bold=True, sz=11, cl=ORANGE, a="c")
    for i, ln in enumerate(["Q(s, a₁)", "Q(s, a₂)", "  ⋮", "Q(s, a|A|)"]):
        d.tx(OUT_X + OUT_W // 2, ym + 20 - i * 18, ln, sz=9, cl=ORANGE, a="c")

    # labels below
    LY = ym - 80
    d.tx(IN_X + IN_W // 2,  LY, "Pixels",          italic=True, sz=9, cl=PURPLE, a="c")
    d.tx(CX_s + 1.5 * (CW + 18), LY, "CNN Feature\nExtraction", italic=True, sz=9, cl=BLUE, a="c")
    d.tx(FC_X + FC_W // 2,  LY, "Reasoning",       italic=True, sz=9, cl=GREEN,  a="c")
    d.tx(OUT_X + OUT_W // 2, LY, "Q-values\nper action", italic=True, sz=9, cl=ORANGE, a="c")

    # ── 16. Input Preprocessing ───────────────────────────────────
    d.page(); d.hdr("Input Preprocessing (Atari)", S5)
    y = CT
    d.tx(ML, y, "Raw Atari frames are preprocessed before entering the CNN.", sz=13)
    y -= 24

    steps16 = [
        ("1. Greyscale",    "210×160 RGB  →  210×160 greyscale   (channels: 3 → 1)"),
        ("2. Crop & Resize","Resize to 84×84 pixels  (standardise spatial resolution)"),
        ("3. Frame Stack",  "Stack 4 consecutive frames → 84×84×4  (captures motion & velocity)"),
        ("4. Normalise",    "Scale pixel values to [0, 1]  (stabilises gradient flow)"),
    ]
    for i, (step, desc) in enumerate(steps16):
        by = y - 20 - i * 66
        d.rb(ML, by - 54, W - 2 * ML, 58, LGRY, r=5)
        d.rb(ML, by - 54, 140, 58, NAVY, r=5)
        d.tx(ML + 8, by - 22, step, bold=True, sz=10, cl=white)
        d.tx(ML + 155, by - 22, desc, sz=11)

    y -= 22 + 4 * 66 + 6
    d.rb(ML, y - 36, W - 2 * ML, 42, LYEL, sk=GOLD, r=5)
    d.tx(ML + 14, y - 12,
         "4 stacked frames give the network a sense of velocity and direction.", sz=12)
    d.tx(ML + 14, y - 28,
         "Without stacking the agent cannot tell if a ball is moving left or right.",
         sz=11, cl=GRAY)

    # ══════════════════════════════════════════════════════════════
    #  SECTION 6 — DQN COMPONENTS
    # ══════════════════════════════════════════════════════════════
    S6 = "DQN Components"

    # ── 17. The Two Core Problems ─────────────────────────────────
    d.page(); d.hdr("The Two Core Problems DQN Solves", S6)
    y = CT
    d.tx(ML, y, "Adding a NN to Q-learning introduces two new training problems:", sz=13)
    y -= 26

    hw17 = (W - 2 * ML - 20) // 2
    d.rb(ML, y - 168, hw17, 168, LRED, sk=RED, r=8)
    d.tx(ML + hw17 // 2, y - 22, "Problem 1", bold=True, sz=12, cl=RED, a="c")
    d.tx(ML + hw17 // 2, y - 40, "Correlated Data", bold=True, sz=14, cl=RED, a="c")
    for i, ln in enumerate(["Consecutive (s,a,r,s') are correlated.",
                             "NNs need i.i.d. data.",
                             "→ training diverges",
                             "",
                             "Fix: Experience Replay"]):
        d.tx(ML + 14, y - 70 - i * 24, ln, bold=(ln.startswith("Fix")), sz=11,
             cl=RED if ln.startswith("Fix") else BODY)

    rx17 = ML + hw17 + 20
    d.rb(rx17, y - 168, hw17, 168, LORG, sk=ORANGE, r=8)
    d.tx(rx17 + hw17 // 2, y - 22, "Problem 2", bold=True, sz=12, cl=ORANGE, a="c")
    d.tx(rx17 + hw17 // 2, y - 40, "Moving Targets", bold=True, sz=14, cl=ORANGE, a="c")
    for i, ln in enumerate(["Target r+γmaxQ(s';θ) shifts",
                             "every gradient step.",
                             "→ training diverges",
                             "",
                             "Fix: Target Network"]):
        d.tx(rx17 + 14, y - 70 - i * 24, ln, bold=(ln.startswith("Fix")), sz=11,
             cl=ORANGE if ln.startswith("Fix") else BODY)

    y -= 184
    d.rb(ML, y - 34, W - 2 * ML, 40, LGRN, sk=GREEN, r=5)
    d.tx(ML + 14, y - 10,
         "✓  DQN solves both with Experience Replay and a frozen Target Network.", sz=12, cl=GREEN)

    # ── 18. Experience Replay ─────────────────────────────────────
    d.page(); d.hdr("Component 1: Experience Replay", S6)
    y = CT
    d.tx(ML, y, "Store all transitions in a buffer; train on random mini-batches.", sz=13)
    y -= 24

    # Buffer diagram
    BUF_H = 80; BUF_Y = y - BUF_H - 10
    d.rb(ML, BUF_Y, W - 2 * ML, BUF_H, LGRY, sk=GRAY, r=6)
    d.tx(ML + 10, BUF_Y + BUF_H - 14, "Replay Buffer  D  (capacity N)",
         bold=True, sz=9, cl=GRAY)
    slots18 = ["(s₁,a₁,r₁,s₁')", "(s₂,a₂,r₂,s₂')", "(s₃,a₃,r₃,s₃')", " … ",
               "(sₙ,aₙ,rₙ,sₙ')"]
    SW = 76; SX = ML + 10; SY = BUF_Y + 12
    scols = [LBLUE, LBLUE, LBLUE, LGRY, LORG]
    for j, (sl, sc) in enumerate(zip(slots18, scols)):
        d.rb(SX + j * (SW + 5), SY, SW, 42, sc, r=4)
        d.tx(SX + j * (SW + 5) + SW // 2, SY + 20, sl, sz=8,
             cl=GRAY if sc == LGRY else BODY, a="c")
    # highlight sampled slots
    for j in [0, 2, 4]:
        cv.setStrokeColor(ORANGE); cv.setLineWidth(2)
        cv.roundRect(SX + j * (SW + 5) - 2, SY - 2, SW + 4, 46, 4, fill=0, stroke=1)

    y = BUF_Y - 16
    d.tx(ML + 10, y - 8, "Mini-batch sampled randomly (highlighted above)",
         italic=True, sz=10, cl=ORANGE)

    y -= 30
    for ln in ["Breaks temporal correlations — training data is i.i.d.",
               "Each transition can be replayed many times (data efficiency)",
               "Stabilises the training distribution over time"]:
        y = d.bul(ML, y, ln, sz=12, cl=GREEN)

    # ── 19. Target Network ────────────────────────────────────────
    d.page(); d.hdr("Component 2: Target Network", S6)
    y = CT
    d.tx(ML, y, "Keep a frozen copy of the network to compute stable TD targets.", sz=13)
    y -= 26

    ONX = ML + 40; ONY = y - 160; ONW = 210; ONH = 120
    TNX = ONX + ONW + 160; TNY = ONY; TNW = 210; TNH = 120

    d.rb(ONX, ONY, ONW, ONH, LGRN, sk=GREEN, r=8)
    d.tx(ONX + ONW // 2, ONY + ONH - 18, "Online Network", bold=True, sz=13, cl=GREEN, a="c")
    d.tx(ONX + ONW // 2, ONY + ONH // 2,  "θ  (updated every step)", sz=10, cl=GREEN, a="c")
    d.tx(ONX + ONW // 2, ONY + 20,        "Q(s, a ; θ)", bold=True, sz=12, cl=GREEN, a="c")

    d.rb(TNX, TNY, TNW, TNH, LORG, sk=ORANGE, r=8)
    d.tx(TNX + TNW // 2, TNY + TNH - 18, "Target Network", bold=True, sz=13, cl=ORANGE, a="c")
    d.tx(TNX + TNW // 2, TNY + TNH // 2,  "θ⁻  (frozen, synced every C steps)",
         sz=9, cl=ORANGE, a="c")
    d.tx(TNX + TNW // 2, TNY + 20,        "Q(s', a ; θ⁻)", bold=True, sz=12, cl=ORANGE, a="c")

    MID19 = ONY + ONH // 2
    d.arr(ONX + ONW + 4, MID19, TNX - 4, MID19, cl=BLUE, lw=2, dash=[5, 4])
    d.tx((ONX + ONW + TNX) // 2, MID19 + 10, "copy θ → θ⁻  every C steps",
         bold=True, sz=10, cl=BLUE, a="c")

    y2 = ONY - 18
    d.eqn(ML, y2 - 50, W - 2 * ML, 56,
          "Target  yᵢ  =  rᵢ  +  γ · maxₐ Q(sᵢ', a ; θ⁻)",
          "θ⁻ is frozen during C gradient steps — provides stable regression targets")

    # ── 20. Epsilon-Greedy ────────────────────────────────────────
    d.page(); d.hdr("Component 3: ε-Greedy Exploration", S6)
    y = CT
    d.tx(ML, y, "Balance exploration with exploitation during training.", sz=13)
    y -= 28
    d.eqn(ML, y - 58, W - 2 * ML, 64,
          ["π(s) = random action          with probability ε",
           "π(s) = argmaxₐ Q(s, a ; θ)  with probability 1 − ε"],
          "ε anneals linearly: εstart (1.0) → εend (0.1) over M training steps")
    y -= 76

    # decay bar
    BAR_W = W - 2 * ML; BAR_H = 26; BAR_Y = y - BAR_H - 6
    d.rb(ML, BAR_Y, BAR_W, BAR_H, LGRY, r=4)
    explore_w = int(BAR_W * 0.82)
    d.rb(ML, BAR_Y, explore_w, BAR_H, LRED, r=4)
    d.tx(ML + 10, BAR_Y + 9, "Exploration  ε: 1.0 → 0.1", bold=True, sz=9, cl=RED)
    d.rb(ML + explore_w + 2, BAR_Y, BAR_W - explore_w - 4, BAR_H, LGRN, r=4)
    d.tx(ML + explore_w + 10, BAR_Y + 9, "Exploit", bold=True, sz=9, cl=GREEN)

    y = BAR_Y - 20
    y = d.bul(ML, y, "High ε early: agent explores freely, populates the replay buffer", sz=12)
    y = d.bul(ML, y, "ε decays over 1M frames in the original DQN paper", sz=12)
    y = d.bul(ML, y, "Low ε later: agent exploits its learned policy", sz=12)

    # ── 21. Loss Function ─────────────────────────────────────────
    d.page(); d.hdr("Component 4: The Loss Function", S6)
    y = CT
    d.tx(ML, y, "Minimise squared Bellman error using sampled mini-batches.", sz=13)
    y -= 30
    d.eqn(ML, y - 66, W - 2 * ML, 72,
          ["L(θ) = E(s,a,r,s')~D  [ (yᵢ − Q(s, a ; θ))² ]",
           "yᵢ  =  r  +  γ · maxₐ' Q(s', a' ; θ⁻)    (stop gradient!)"],
          "Gradient ∇θ L flows only through Q(s,a;θ) — NOT through yᵢ")
    y -= 88

    d.tx(ML, y, "In practice:", bold=True, sz=12)
    y -= 20
    y = d.bul(ML, y, "Huber loss instead of MSE — more robust to large TD errors", sz=12)
    y = d.bul(ML, y, "Clip rewards to [−1, +1] to normalise across different games", sz=12)
    y = d.bul(ML, y, "Update online θ every step;  copy to θ⁻ every C ≈ 1000 steps", sz=12)
    y -= 14
    d.rb(ML, y - 34, W - 2 * ML, 40, LYEL, sk=GOLD, r=5)
    d.tx(ML + 14, y - 10, "⚠  Never backpropagate through the target yᵢ — it must be treated as a constant.", sz=12)

    # ══════════════════════════════════════════════════════════════
    #  SECTION 7 — END-TO-END FLOW
    # ══════════════════════════════════════════════════════════════
    S7 = "End-to-End Flow"

    # ── 22. Algorithm ─────────────────────────────────────────────
    d.page(); d.hdr("DQN Training Algorithm", S7)

    pseudo = [
        ("Initialise",  "replay buffer D (capacity N),  online network Q(θ),  target θ⁻ ← θ"),
        ("",            ""),
        ("For episode",  "1 … M:"),
        ("  Observe",   "initial state s₁"),
        ("  For t",      "= 1 … T:"),
        ("    Select",  "aₜ: random w.p. ε,  else argmaxₐ Q(sₜ,a;θ)    [ε-greedy]"),
        ("    Execute", "aₜ  →  observe rₜ, sₜ₊₁"),
        ("    Store",   "(sₜ, aₜ, rₜ, sₜ₊₁) in D"),
        ("    Sample",  "mini-batch {(sᵢ, aᵢ, rᵢ, sᵢ')} from D"),
        ("    Compute", "yᵢ = rᵢ + γ maxₐ Q(sᵢ',a;θ⁻)  [terminal: yᵢ = rᵢ]"),
        ("    Step",    "gradient descent on  L(θ) = Σ (yᵢ − Q(sᵢ,aᵢ;θ))²"),
        ("    Every C", "θ⁻ ← θ"),
    ]
    code_h = len(pseudo) * 17 + 22
    code_y = CT - code_h - 2
    d.rb(ML, code_y, W - 2 * ML, code_h, LGRY, sk=GRAY, r=6)
    cy22 = code_y + code_h - 18
    for kw, rest in pseudo:
        if not kw:
            cy22 -= 5; continue
        ind = (len(kw) - len(kw.lstrip())) * 2
        kw_s = kw.lstrip()
        col22 = BLUE if ind == 0 else (ORANGE if ind == 4 else PURPLE)
        d.tx(ML + 12 + ind, cy22, kw_s, bold=True, sz=10, cl=col22)
        rest_x = ML + 12 + ind + 6 * len(kw_s) + 6   # rough fixed width
        d.tx(ML + 12 + ind + 90, cy22, rest, sz=10)
        cy22 -= 17

    # ── 23. Data Flow ─────────────────────────────────────────────
    d.page(); d.hdr("Data Flow: One Training Step", S7)

    steps23 = [
        ("Environment",      "Produces sₜ, rₜ",        LGRN,  GREEN),
        ("Replay\nBuffer",   "Stores (s,a,r,s')",       LBLUE, BLUE),
        ("Mini-batch\nSample","32 random transitions",  LPUR,  PURPLE),
        ("Online Net\nQ(s,a;θ)","Predicts Q-values",   LORG,  ORANGE),
        ("Loss &\nBackprop", "Updates θ",               LRED,  RED),
    ]
    N23 = len(steps23); SW23 = 128; SH23 = 82; GAP23 = 22
    TW23 = N23 * SW23 + (N23 - 1) * GAP23
    SX23 = (W - TW23) // 2; SY23 = H // 2 - SH23 // 2 - 16

    for i, (name, desc, bc, tc) in enumerate(steps23):
        bx = SX23 + i * (SW23 + GAP23)
        d.rb(bx, SY23, SW23, SH23, bc, sk=tc, r=6)
        for j, ln in enumerate(name.split("\n")):
            d.tx(bx + SW23 // 2, SY23 + SH23 - 18 - j * 14, ln, bold=True, sz=10, cl=tc, a="c")
        d.tx(bx + SW23 // 2, SY23 + 10, desc, sz=8, cl=GRAY, a="c")
        if i < N23 - 1:
            d.arr(bx + SW23 + 2, SY23 + SH23 // 2, bx + SW23 + GAP23 - 2, SY23 + SH23 // 2,
                  cl=GRAY, lw=1.5)

    # target network above step 4
    tn_bx = SX23 + 3 * (SW23 + GAP23) - 10
    tn_by = SY23 + SH23 + 30
    tn_bw = SW23 + 20; tn_bh = 58
    d.rb(tn_bx, tn_by, tn_bw, tn_bh, LORG, sk=ORANGE, r=6)
    d.tx(tn_bx + tn_bw // 2, tn_by + 40, "Target Net θ⁻",  bold=True, sz=10, cl=ORANGE, a="c")
    d.tx(tn_bx + tn_bw // 2, tn_by + 22, "Q(s',a ; θ⁻)",  bold=True, sz=10, cl=ORANGE, a="c")
    d.tx(tn_bx + tn_bw // 2, tn_by + 8,  "frozen",        italic=True, sz=8,  cl=GRAY,   a="c")
    loss_bx = SX23 + 4 * (SW23 + GAP23)
    d.arr(tn_bx + tn_bw // 2, tn_by, loss_bx + SW23 // 2, SY23 + SH23,
          cl=ORANGE, lw=1.5, dash=[4, 3])
    # sync arrow
    d.arr(loss_bx + SW23 // 2, SY23 + SH23 + 4, tn_bx + tn_bw // 2, tn_by + tn_bh,
          cl=BLUE, lw=1, dash=[4, 3])
    d.tx(loss_bx + SW23 // 2 + 12, SY23 + SH23 + 20,
         "copy θ→θ⁻\nevery C steps", sz=8, cl=BLUE)

    # ── 24. Putting It Together ────────────────────────────────────
    d.page(); d.hdr("Putting It All Together", S7)
    y = CT
    d.tx(ML, y, "Each training step executes six operations in order:", sz=13)
    y -= 24

    flow24 = [
        (ACT,   "① Act",    "ε-greedy selects action aₜ from online network Q(s,a;θ)"),
        (GREEN, "② Store",  "Transition (sₜ,aₜ,rₜ,sₜ₊₁) appended to replay buffer D"),
        (PURPLE,"③ Sample", "Random mini-batch of 32 transitions drawn from D"),
        (ORANGE,"④ Target", "Frozen θ⁻ computes  yᵢ = r + γ maxQ(s';θ⁻)"),
        (RED,   "⑤ Update", "Gradient step minimises (yᵢ − Q(sᵢ,aᵢ;θ))²  →  update θ"),
        (BLUE,  "⑥ Sync",   "Every C steps: copy online weights  θ⁻ ← θ"),
    ]
    for tc24, label, desc in flow24:
        d.rb(ML, y - 30, W - 2 * ML, 32, LGRY, r=4)
        d.rb(ML, y - 30, 5, 32, tc24, r=2)
        d.tx(ML + 14, y - 12, label, bold=True, sz=11, cl=tc24)
        d.tx(ML + 14 + 58, y - 12, desc, sz=11)
        y -= 38

    # ══════════════════════════════════════════════════════════════
    #  SECTION 8 — EXAMPLE
    # ══════════════════════════════════════════════════════════════
    S8 = "Example: CartPole"

    # ── 25. CartPole Environment ──────────────────────────────────
    d.page(); d.hdr("Example: CartPole-v1", S8)
    y = CT
    d.tx(ML, y, "Balance a pole on a moving cart — a classic RL benchmark.", sz=13)
    y -= 26

    # CartPole sketch (right side)
    CCX = 710; CCY = 300
    cv.setStrokeColor(GRAY); cv.setLineWidth(2)
    cv.line(CCX - 110, CCY - 20, CCX + 110, CCY - 20)
    d.rb(CCX - 35, CCY - 20, 70, 28, NAVY, r=4)
    d.tx(CCX, CCY - 4, "cart", bold=True, sz=9, cl=white, a="c")
    cv.setFillColor(MGRY)
    cv.circle(CCX - 18, CCY - 20, 8, fill=1, stroke=0)
    cv.circle(CCX + 18, CCY - 20, 8, fill=1, stroke=0)
    pole_l = 88; pole_a = math.radians(10)
    px25 = CCX + pole_l * math.sin(pole_a); py25 = CCY - 20 + pole_l * math.cos(pole_a)
    cv.setStrokeColor(ORANGE); cv.setLineWidth(5)
    cv.line(CCX, CCY - 20, px25, py25)
    cv.setFillColor(ORANGE); cv.circle(px25, py25, 7, fill=1, stroke=0)

    # Info panel (left side)
    info25 = [
        ("State",     "4D: [cart x, cart ẋ, pole θ, pole θ̇]"),
        ("Actions",   "2 discrete: push left / push right"),
        ("Reward",    "+1 for every step the pole stays upright"),
        ("Goal",      "Survive 500 steps (solved threshold)"),
        ("Network",   "4 inputs → FC(64, ReLU) → FC(64, ReLU) → 2 Q-values"),
    ]
    IW = 560
    for j, (k, v) in enumerate(info25):
        by25 = y - 18 - j * 42
        d.rb(ML, by25 - 30, IW, 36, LGRY, r=4)
        d.tx(ML + 10, by25 - 12, k + ":", bold=True, sz=11, cl=BLUE)
        d.tx(ML + 100, by25 - 12, v, sz=10)

    # ── 26. DQN on CartPole ───────────────────────────────────────
    d.page(); d.hdr("DQN Applied to CartPole: Step by Step", S8)
    y = CT
    d.tx(ML, y, "One full training interaction traced through DQN:", sz=13)
    y -= 22

    steps26 = [
        ("Observe",   "s = [0.02, −0.04, 0.01, 0.03]"),
        ("ε-greedy",  "ε=0.9  →  random action: push right (a=1)"),
        ("Transition","r=1.0,  s'=[0.03, 0.15, 0.03, −0.26]  (pole still upright)"),
        ("Buffer",    "Store (s, 1, 1.0, s') in D.  Buffer reaches 32 → begin sampling"),
        ("Target",    "y = 1.0 + 0.99 × maxQ(s'; θ⁻) = 1.0 + 0.99×0.54 ≈ 1.53"),
        ("Online",    "Q(s, 1; θ) = 0.31"),
        ("Loss",      "L = (1.53 − 0.31)² = 1.49  →  backprop  →  θ updated"),
        ("Repeat",    "Over 50k steps ε anneals 1.0→0.1; agent learns to balance"),
    ]
    for i, (step, desc) in enumerate(steps26):
        by26 = y - 10 - i * 40
        d.rb(ML, by26 - 30, W - 2 * ML, 34, LGRY, r=4)
        d.rb(ML, by26 - 30, 82, 34, NAVY, r=4)
        d.tx(ML + 8, by26 - 10, step, bold=True, sz=10, cl=white)
        d.tx(ML + 96, by26 - 10, desc, sz=10)

    # ══════════════════════════════════════════════════════════════
    #  SECTION 9 — VARIANTS
    # ══════════════════════════════════════════════════════════════
    S9 = "DQN Variants"

    # ── 27. Double DQN ────────────────────────────────────────────
    d.page(); d.hdr("Double DQN (DDQN)", S9)
    y = CT
    d.tx(ML, y, "DQN overestimates Q-values — the same network selects and evaluates.", sz=13)
    y -= 22

    d.rb(ML, y - 50, W - 2 * ML, 54, LRED, sk=RED, r=5)
    d.tx(ML + 14, y - 16, "DQN target:", bold=True, sz=12, cl=RED)
    d.tx(ML + 130, y - 16,
         "yᵢ = rᵢ + γ  maxₐ  Q(sᵢ', a ; θ⁻)     ← same net selects & evaluates",
         sz=11, cl=RED)
    d.tx(ML + 14, y - 34, "Problem: maximisation bias → systematic overestimation of Q-values",
         sz=10, cl=RED)

    y -= 66
    d.rb(ML, y - 50, W - 2 * ML, 54, LGRN, sk=GREEN, r=5)
    d.tx(ML + 14, y - 16, "DDQN target:", bold=True, sz=12, cl=GREEN)
    d.tx(ML + 140, y - 16,
         "yᵢ = rᵢ + γ  Q(sᵢ', argmaxₐ Q(sᵢ',a ; θ) ; θ⁻)",
         sz=11, cl=GREEN)
    d.tx(ML + 14, y - 34, "Online θ selects,  target θ⁻ evaluates  →  decoupled, less biased",
         sz=10, cl=GREEN)

    y -= 66
    d.tx(ML, y, "Key change:", bold=True, sz=12)
    y -= 20
    y = d.bul(ML, y, "Online network θ:   'Which action looks best from s'?'", sz=12)
    y = d.bul(ML, y, "Target network θ⁻:  'What is that action actually worth?'", sz=12)
    y = d.bul(ML, y, "Tiny code change — large improvement in value estimates", sz=12)

    # ── 28. Dueling DQN ───────────────────────────────────────────
    d.page(); d.hdr("Dueling Network Architecture", S9)
    y = CT
    d.tx(ML, y, "Decompose Q(s,a) into state value V(s) and advantage A(s,a).", sz=13)
    y -= 26

    # Shared block
    SHX = ML + 20; SHY = y - 100; SHW = 190; SHH = 80
    d.rb(SHX, SHY, SHW, SHH, LGRY, sk=GRAY, r=6)
    d.tx(SHX + SHW // 2, SHY + 50, "Shared CNN", bold=True, sz=12, cl=GRAY, a="c")
    d.tx(SHX + SHW // 2, SHY + 30, "Feature", sz=11, cl=GRAY, a="c")
    d.tx(SHX + SHW // 2, SHY + 14, "Extractor", sz=11, cl=GRAY, a="c")

    MID_SH = SHY + SHH // 2
    # V stream
    VX = SHX + SHW + 60; VY = y - 66; VW = 130; VH = 46
    d.arr(SHX + SHW, MID_SH, VX, VY + VH // 2 + 5, cl=BLUE, lw=1.5)
    d.rb(VX, VY, VW, VH, LBLUE, sk=BLUE, r=6)
    d.tx(VX + VW // 2, VY + 30, "Value Stream", bold=True, sz=10, cl=BLUE, a="c")
    d.tx(VX + VW // 2, VY + 12, "V(s)", bold=True, sz=14, cl=BLUE, a="c")

    # A stream
    AX = VX; AY = VY - 76; AW = 130; AH = 46
    d.arr(SHX + SHW, MID_SH, AX, AY + AH // 2 - 5, cl=PURPLE, lw=1.5)
    d.rb(AX, AY, AW, AH, LPUR, sk=PURPLE, r=6)
    d.tx(AX + AW // 2, AY + 30, "Advantage Stream", bold=True, sz=10, cl=PURPLE, a="c")
    d.tx(AX + AW // 2, AY + 12, "A(s,a)", bold=True, sz=14, cl=PURPLE, a="c")

    # Combine
    CMX = VX + VW + 40; CMY = AY - 10; CMW = 170; CMH = VY + VH - AY + 20
    d.arr(VX + VW, VY + VH // 2, CMX, CMY + CMH * 2 // 3, cl=BLUE, lw=1.5)
    d.arr(AX + AW, AY + AH // 2, CMX, CMY + CMH // 3, cl=PURPLE, lw=1.5)
    d.rb(CMX, CMY, CMW, CMH, LORG, sk=ORANGE, r=6)
    d.tx(CMX + CMW // 2, CMY + CMH - 18, "Combine", bold=True, sz=11, cl=ORANGE, a="c")
    d.tx(CMX + CMW // 2, CMY + CMH // 2,
         "Q = V + A − mean(A)", sz=10, cl=ORANGE, a="c")
    d.tx(CMX + CMW // 2, CMY + 14, "Q(s,a)", bold=True, sz=13, cl=ORANGE, a="c")

    y2 = CMY - 18
    d.rb(ML, y2 - 36, W - 2 * ML, 42, LYEL, sk=GOLD, r=5)
    d.tx(ML + 14, y2 - 14,
         "Benefit: network learns V(s) even for actions that don't matter → faster convergence.", sz=12)

    # ── 29. PER + Noisy + Rainbow ─────────────────────────────────
    d.page(); d.hdr("Prioritised Replay · Noisy Nets · Rainbow", S9)
    y = CT

    vs29 = [
        ("Prioritised\nExperience\nReplay (PER)",
         ["Sample high-TD-error transitions more",
          "Learn more from surprising experiences",
          "IS weights correct sampling bias"],
         ORANGE, LORG),
        ("Noisy DQN",
         ["Learnable noise in FC layer weights",
          "State-dependent exploration (smarter)",
          "Replaces ε-greedy schedule entirely"],
         PURPLE, LPUR),
        ("Rainbow\n(2017)",
         ["Combines 6 DQN improvements:",
          "DDQN + Dueling + PER +",
          "Noisy + Distributional + n-step",
          "State of the art on Atari (at time)"],
         TEAL, LTEAL),
    ]
    bw29 = (W - 2 * ML - 40) // 3
    for i, (name, pts, tc, bc) in enumerate(vs29):
        bx = ML + i * (bw29 + 20); by = y - 220
        d.rb(bx, by, bw29, 220, bc, sk=tc, r=8)
        name_lines = name.split("\n")
        d.rb(bx, by + 220 - 26 * len(name_lines), bw29, 26 * len(name_lines), tc, r=8)
        for j, nl in enumerate(name_lines):
            d.tx(bx + bw29 // 2,
                 by + 220 - 16 - j * 20,
                 nl, bold=True, sz=11, cl=white, a="c")
        for k, pt in enumerate(pts):
            d.tx(bx + 10, by + 160 - k * 32, "▸", sz=10, cl=tc)
            d.tx(bx + 22, by + 160 - k * 32, pt, sz=9)

    # ══════════════════════════════════════════════════════════════
    #  SECTION 10 — CONCLUSION
    # ══════════════════════════════════════════════════════════════
    S10 = "Conclusion"

    # ── 30 (final). Key Takeaways ─────────────────────────────────
    d.page(); d.hdr("Key Takeaways", S10)
    y = CT
    takes = [
        (BLUE,   "DQN = Q-Learning + Deep NN + Experience Replay + Target Network"),
        (GREEN,  "Experience Replay: random mini-batches break temporal correlations"),
        (ORANGE, "Target Network: frozen copy stops the agent chasing a moving target"),
        (PURPLE, "ε-greedy exploration anneals from 1.0 → 0.1 over training"),
        (RED,    "Loss = Huber((r + γ maxQ(s';θ⁻)) − Q(s,a;θ))²  with stop-gradient"),
        (TEAL,   "DDQN, Dueling, PER, Rainbow each fix a specific remaining weakness"),
    ]
    for i, (tc, text) in enumerate(takes):
        by_t = y - 10 - i * 56
        d.rb(ML, by_t - 42, W - 2 * ML, 46, LGRY, r=5)
        d.rb(ML, by_t - 42, 6, 46, tc, r=2)
        d.tx(ML + 20, by_t - 16, text, bold=True, sz=12)

    # ── END SLIDE ─────────────────────────────────────────────────
    # (30 slides total — title is slide 1, takeaways is slide 30)
    # If you want a final blank/thank-you, increment TOTAL and add d.page() here.

    d.save()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dqn_lecture.pdf")
    print(f"Generating  {out} …")
    build(out)
    print("Done.")
