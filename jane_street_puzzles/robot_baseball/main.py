import math
import numpy as np

def solve_game_matrix(a,b,c,d):
    # rows: Ball, Strike; cols: Wait, Swing
    row_min0 = min(a, b)
    row_min1 = min(c, d)
    maximin = max(row_min0, row_min1)
    col_max0 = max(a, c)
    col_max1 = max(b, d)
    minimax = min(col_max0, col_max1)
    x = 0.0
    y = 1.0
    if abs(maximin - minimax) < 1e-15:
        V = maximin
        # pick simple pure actions to represent the saddle
        x = 1.0 if max(a,c) == V else 0.0
        y = 1.0 if row_min0 == V else 0.0
    else:
        denom = (a - b - c + d)
        if abs(denom) < 1e-18:
            V = maximin
            x = 1.0 if a>=b else 0.0
            y = 1.0 if row_min0==maximin else 0.0
        else:
            x = (d - b) / denom
            x = max(0.0, min(1.0, x))
            V = x*a + (1.0 - x)*b
            denom_y = denom
            if abs(denom_y) < 1e-18:
                y = 1.0 if col_max0 <= col_max1 else 0.0
            else:
                y = (d - c) / denom_y
                y = max(0.0, min(1.0, y))
    return V, x, y

def compute_for_p(p):
    V = [[0.0]*3 for _ in range(4)]
    batter_mix = [[0.0]*3 for _ in range(4)]
    pitcher_mix = [[0.0]*3 for _ in range(4)]

    # backward recursion: compute states with larger b or s first
    for b in range(3, -1, -1):
        for s in range(2, -1, -1):
            # set up expected-score matrix entries
            a = 1.0 if (b+1)==4 else V[b+1][s]           # Ball + Wait -> walk or (b+1,s)
            b_entry = 0.0 if (s+1)>=3 else V[b][s+1]     # Ball + Swing -> strike or (b,s+1)
            c = 0.0 if (s+1)>=3 else V[b][s+1]           # Strike + Wait
            next_val = 0.0 if (s+1)>=3 else V[b][s+1]
            d = p*4.0 + (1.0-p)*next_val                 # Strike + Swing: HR with prob p (4 pts)
            V_val, x, y = solve_game_matrix(a, b_entry, c, d)
            V[b][s] = V_val
            batter_mix[b][s] = x
            pitcher_mix[b][s] = y

    # compute full-count probability R under the equilibrium mixes
    R = [[0.0]*3 for _ in range(4)]
    R[3][2] = 1.0
    for b in range(3, -1, -1):
        for s in range(2, -1, -1):
            if b==3 and s==2:
                continue
            x = batter_mix[b][s]
            y = pitcher_mix[b][s]
            prob = 0.0
            # Ball, Wait -> (b+1,s) or walk
            prob += y * x * (1.0 if (b+1==3 and s==2) else (0.0 if b+1>=4 else R[b+1][s]))
            # Ball, Swing -> (b,s+1) or strikeout
            prob += y * (1.0 - x) * (1.0 if (b==3 and s+1==2) else (0.0 if s+1>=3 else R[b][s+1]))
            # Strike, Wait -> (b,s+1) or strikeout
            prob += (1.0 - y) * x * (1.0 if (b==3 and s+1==2) else (0.0 if s+1>=3 else R[b][s+1]))
            # Strike, Swing -> HR w.p. p (terminal, not full count) else (b,s+1)
            prob += (1.0 - y) * (1.0 - x) * ((1.0 - p) * (1.0 if (b==3 and s+1==2) else (0.0 if s+1>=3 else R[b][s+1])))
            R[b][s] = prob

    return V, batter_mix, pitcher_mix, R

def q_of_p(p):
    return compute_for_p(p)[3][0][0]

# Golden-section search on [0,0.5]
def golden_maximize(a,b,func,iters=40):
    gr = (math.sqrt(5) - 1) / 2
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = func(c); fd = func(d)
    for _ in range(iters):
        if fc > fd:
            b = d; d = c; fd = fc
            c = b - gr * (b - a); fc = func(c)
        else:
            a = c; c = d; fc = fd
            d = a + gr * (b - a); fd = func(d)
    p_opt = 0.5*(a+b)
    return p_opt, func(p_opt)

p_opt, q_opt = golden_maximize(0.0, 0.5, q_of_p, iters=40)
print("p_opt (10 dp) =", f"{p_opt:.10f}")
print("q_opt (10 dp) =", f"{q_opt:.10f}")
