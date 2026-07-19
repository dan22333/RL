### MDP Value Iteration and Policy Iteration — VECTORIZED / OPTIMIZED
###
### Same math as vi_and_pi.py, but every Python `for` loop over states is
### replaced by a single batched tensor operation. Each function has the
### equivalent PyTorch one-liner in a comment so you can see "the torch way".
###
### Key trick used everywhere:
###   Q(s,a) = R(s,a) + gamma * sum_{s'} T(s,a,s') * V(s')
###   With R:(S,A), T:(S,A,S'), V:(S',):   Q = R + gamma * (T @ V)
###   `T @ V` is a batched matrix-vector product: (S,A,S') · (S',) -> (S,A).
###   That single line computes the backup for ALL states and actions at once.

import numpy as np
from riverswim import RiverSwim

np.set_printoptions(precision=3)


def q_values(R, T, gamma, V):
    """
    Full Q-table for every (state, action) in one shot — the vectorized backup.

    R: (S, A)   T: (S, A, S')   V: (S',)   ->   Q: (S, A)

    This replaces the per-(state,action) `bellman_backup` AND the inner
    `for i in range(num_states)` loop from the original file.

    torch:  Q = R + gamma * torch.matmul(T, V)          # or einsum('sap,p->sa', T, V)
    """
    return R + gamma * (T @ V)          # (S,A,S') @ (S',) -> (S,A)


def bellman_backup(state, action, R, T, gamma, V):
    """Single (state, action) backup — kept for API compatibility. Just one cell of q_values."""
    return R[state, action] + gamma * (T[state, action] @ V)


def policy_evaluation(policy, R, T, gamma, tol=1e-3, direct=True):
    """
    Value of a fixed policy, V^pi.

    Two ways:
      direct=True  -> solve the linear system exactly, NO iteration (the real optimization).
      direct=False -> iterate the pi-backup, but vectorized (no python state loop).

    policy: (S,)  ->  V^pi: (S,)
    """
    num_states, num_actions = R.shape
    idx = np.arange(num_states)

    # Gather the reward/transition rows for the action the policy picks in each state.
    R_pi = R[idx, policy]               # (S,)      R(s, pi(s))
    P_pi = T[idx, policy]               # (S, S')   p(s'|s, pi(s))

    if direct:
        # V^pi = (I - gamma*P_pi)^{-1} R_pi   — closed form, because B^pi has no max (it's linear).
        # torch:  V = torch.linalg.solve(torch.eye(S) - gamma*P_pi, R_pi)
        return np.linalg.solve(np.eye(num_states) - gamma * P_pi, R_pi)

    # Iterative fallback: V <- R_pi + gamma * P_pi @ V   until it stops moving.
    V = np.zeros(num_states)
    while True:
        V_new = R_pi + gamma * (P_pi @ V)          # whole vector at once, no state loop
        if np.linalg.norm(V_new - V, np.inf) < tol:
            return V_new
        V = V_new


def policy_improvement(policy, R, T, V_policy, gamma):
    """
    Greedy step: best action in every state at once.
    (Already vectorized in the original — this is the same argmax over the Q-table.)

    torch:  policy = q_values(R, T, gamma, V).argmax(dim=1)
    """
    return np.argmax(q_values(R, T, gamma, V_policy), axis=1)


def policy_iteration(R, T, gamma, tol=1e-3):
    """Alternate exact evaluation + greedy improvement until the policy stops changing."""
    num_states, num_actions = R.shape
    policy = np.zeros(num_states, dtype=int)

    n_iter = 0
    while True:
        V_policy = policy_evaluation(policy, R, T, gamma, tol)     # exact linear solve
        new_policy = policy_improvement(policy, R, T, V_policy, gamma)
        n_iter += 1
        if np.array_equal(new_policy, policy):                    # integer policies: exact equality
            break
        policy = new_policy
    print(f'Finish policy iteration in {n_iter} rounds')
    return V_policy, policy


def value_iteration(R, T, gamma, tol=1e-3):
    """
    Repeat the max-backup until V converges, then read the greedy policy ONCE at the end.

    torch (the whole loop body):
        Q = R + gamma * torch.matmul(T, V)
        V = Q.max(dim=1).values
    """
    num_states, num_actions = R.shape
    V = np.zeros(num_states)

    n_iter = 0
    while True:
        n_iter += 1
        Q = q_values(R, T, gamma, V)               # (S,A) — one line, all states/actions
        V_new = np.max(Q, axis=1)                   # (S,)  — max over actions
        if np.linalg.norm(V_new - V, np.inf) < tol:
            V = V_new
            break
        V = V_new

    policy = np.argmax(q_values(R, T, gamma, V), axis=1)   # greedy policy, computed once
    print(f'Finish value iteration in {n_iter} rounds')
    return V, policy


def question_d(seed=1234, tol=1e-9):
    """
    Q4(d): for each current, find the LARGEST 2-decimal gamma at which the optimal
    action in the far-left state (s0) is still LEFT (bank the safe 0.005) rather
    than RIGHT (swim upstream for the +1).

    Method: sweep gamma = 0.01 .. 0.99, solve for the optimal policy, and record
    the largest gamma whose optimal action at s0 is LEFT. The threshold is where
    that action flips LEFT -> RIGHT.
    """
    LEFT, RIGHT = 0, 1
    print("\n" + "-" * 55)
    print(f"{'Current':8s}{'P(forward)':>12s}{'largest gamma LEFT':>22s}")
    print("-" * 55)
    for cur in ['WEAK', 'MEDIUM', 'STRONG']:
        env = RiverSwim(cur, seed)
        R, T = env.get_model()
        p_forward = T[1, RIGHT, 2]                    # forward prob from a middle state
        largest_left = None
        for g100 in range(1, 100):                    # every 2-decimal gamma
            g = g100 / 100
            _, pol = value_iteration(R, T, gamma=g, tol=tol)
            if pol[0] == LEFT:
                largest_left = g                      # keep updating -> ends as the largest
        print(f"{cur:8s}{p_forward:>12.2f}{largest_left:>22.2f}")
    print("-" * 55)


if __name__ == "__main__":
    SEED = 1234
    RIVER_CURRENT = 'WEAK'
    assert RIVER_CURRENT in ['WEAK', 'MEDIUM', 'STRONG']
    env = RiverSwim(RIVER_CURRENT, SEED)

    R, T = env.get_model()
    discount_factor = 0.99

    print("\n" + "-" * 25 + "\nBeginning Policy Iteration\n" + "-" * 25)
    V_pi, policy_pi = policy_iteration(R, T, gamma=discount_factor, tol=1e-3)
    print(V_pi)
    print([['L', 'R'][a] for a in policy_pi])

    print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)
    V_vi, policy_vi = value_iteration(R, T, gamma=discount_factor, tol=1e-3)
    print(V_vi)
    print([['L', 'R'][a] for a in policy_vi])

    # ---- Q4(d): the gamma-threshold experiment ----
    question_d(SEED)
