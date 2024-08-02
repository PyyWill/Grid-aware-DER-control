import cvxpy as cp
import numpy as np

#################### Topo ####################
N = 3 # Bus number
adj_matrix = np.array([
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 0]
]) # Upper triangular matrix
battery_node_ls = [1] # leaf node
PV_node_ls = [2] # leaf node
impedance_node_ls = [] # leaf node

T = 1

#################### Params ####################
gamma = np.array([
    [1, -0.5 + 0.5 * np.sqrt(3) * 1j, -0.5 - 0.5 * np.sqrt(3) * 1j],
    [-0.5 - 0.5 * np.sqrt(3) * 1j, 1, -0.5 + 0.5 * np.sqrt(3) * 1j],
    [-0.5 + 0.5 * np.sqrt(3) * 1j, -0.5 - 0.5 * np.sqrt(3) * 1j, 1],
])
B = np.outer(gamma[:,0], gamma[:,0].conj())

Y_common = np.array([
    [0.1 + 0.3j, 0.05 + 0.2j, 0.05 + 0.25j],
    [0.05 + 0.2j, 0.1 + 0.4j, 0.05 + 0.15j],
    [0.05 + 0.25j, 0.05 + 0.15j, 0.1 + 0.5j]
])

Y = {} # should be a list
connections = np.where(adj_matrix == 1)
for i, j in zip(*connections):
    Y[f"{i}{j}"] = Y_common # Z_ij

RTP = np.array([5,5])

lb = -500 # should be a list
ub = 500 # should be a list

constraints = []
#################### Utils ####################
def dfs(start, visited=None):
    global adj_matrix
    adj_matrix = np.triu(adj_matrix)
    if visited is None:
        visited = set()
    visited.add(start)
    for neighbor, isConnected in enumerate(adj_matrix[start]):
        if isConnected and neighbor not in visited:
            dfs(neighbor, visited)
    return visited

def create_grid_variable(N):
    S = {}; V = {}; I = {}
    s = {}; v = {}; i = {}
    lam = {}; va = {}
    global constraints
    for t in range(T):
        for n in range(N):
            s[f"{n}_{t}"] = cp.Variable((3, 1), complex=True)
            V[f"{n}_{t}"] = cp.Variable((3, 1), complex=True)
            v[f"{n}_{t}"] = cp.Variable((3, 3), complex=True)
            va[f"{n}_{t}"] = cp.Variable(complex=False)
            constraints += [v[f"{n}_{t}"] >> 0]
            constraints += [v[f"{n}_{t}"] - va[f"{n}_{t}"]*B == 0]
        for j,k in zip(*connections):
            S[f"{j}{k}_{t}"] = cp.Variable((3, 3), complex=True)
            I[f"{j}{k}_{t}"] = cp.Variable((3, 1), complex=True)
            i[f"{j}{k}_{t}"] = cp.Variable((3, 3), complex=True)
            lam[f"{j}{k}_{t}"] = cp.Variable((3, 1), complex=True)
            constraints += [i[f"{j}{k}_{t}"] >> 0]
    return (S, V, I, s, v, i, lam, va)

def create_device_variable():
    global battery_node_ls, PV_node_ls, impedance_node_ls
    p_battery = {}; p_PV = {}; p_impedance = {}
    for t in range(T):
        for n in battery_node_ls:
            p_battery[f"{n}_{t}"] = cp.Variable((3, 1))
        for n in PV_node_ls:
            p_PV[f"{n}_{t}"] = cp.Variable((3, 1))
        for n in impedance_node_ls:
            p_impedance[f"{n}_{t}"] = cp.Variable((3, 1))
    return (p_battery, p_PV, p_impedance)

def power_flow_equation(grid_variable):
    S, V, I, s, v, i, lam, va = grid_variable
    global connections, constraints

    # squared equation (v-V, i-I, s-S)
    for t in range(T):
        for j, k in zip(*connections):
            constraints += [S[f"{j}{k}_{t}"]-cp.matmul(V[f"{j}_{t}"], cp.conj(I[f"{j}{k}_{t}"]).T) == 0]
        for n in range(N):
            constraints += [v[f"{n}_{t}"]-cp.matmul(V[f"{n}_{t}"], cp.conj(V[f"{n}_{t}"]).T) == 0]

    return None

def lindistflow_constraints(grid_variable):
    S, V, I, s, v, i, lam, va = grid_variable
    global connections, constraints

    # tree equation
    for t in range(T):
        constraints += [sum(s[f"{n}_{t}"] for n in range(N)) == 0]
    
    # subtree constraints
    for t in range(T):
        for k, j in zip(*connections):
            sub_node = dfs(j)
            constraints += [lam[f"{k}{j}_{t}"] + sum(s[f"{n}_{t}"] for n in sub_node) == 0]
            constraints += [S[f"{k}{j}_{t}"] - cp.matmul(gamma,cp.diag(lam[f"{k}{j}_{t}"])) == 0]
    
    # unique path constraints
    for t in range(T):
        for j, k in zip(*connections):
            if j>k: # for Y_common
                voltage_diff = Y[f"{j}{k}"] @ (v[f"{j}_{t}"]-v[f"{k}_{t}"]) @ cp.conj(Y[f"{j}{k}"]).T
                constraints += [voltage_diff - cp.conj(S[f"{j}{k}_{t}"]).T @ cp.conj(Y[f"{j}{k}"]).T - Y[f"{j}{k}"] @ S[f"{j}{k}_{t}"] == 0]         
    
    return None

def operational_constraints(grid_variable):
    S, V, I, s, v, i, lam, va = grid_variable
    global connections, constraints

    # v, s constraints
    for t in range(T):
        for n in range(N):
            constraints += [cp.abs(cp.diag(v[f"{n}_{t}"])[i]) <= 360 for i in range(3)]
            constraints += [cp.abs(s[f"{n}_{t}"][i]) <= 1000 for i in range(3)]
    # i constraints
    for j, k in zip(*connections):
        constraints += [cp.abs(i[f"{j}{k}_{t}"][p]) <= 100 for p in range(3)]

    return None

def nodal_injection_equation(grid_variable, device_variable):
    S, V, I, s, v, i, lam, va = grid_variable
    p_battery, p_PV, p_impedance = device_variable
    global battery_node_ls, PV_node_ls, impedance_node_ls
    global constraints

    for n in PV_node_ls:
        for t in range(T):
            constraints += [p_PV[f"{n}_{t}"] - 200 + cp.real(s[f"{n}_{t}"]) == 0]
            # PV generation uncertainty
            constraints += [p_PV[f"{n}_{t}"] <= 20]
            constraints += [p_PV[f"{n}_{t}"] >= 0]
    for n in impedance_node_ls:
        for t in range(T):
            constraints += [p_impedance[f"{n}_{t}"] - 200 + cp.real(s[f"{n}_{t}"]) == 0]
    for n in battery_node_ls:
        constraints += [sum(p_battery[f"{n}_{t}"] for t in range(T)) <= 1000] # SOC constraints
        for t in range(T):
            constraints += [p_battery[f"{n}_{t}"] - 200 + cp.real(s[f"{n}_{t}"]) == 0]
            constraints += [p_battery[f"{n}_{t}"] <= 10]
            constraints += [p_battery[f"{n}_{t}"] >= -10]

    return None
#################### Variables ####################
grid_variable = create_grid_variable(N)
S, V, I, s, v, i, lam, va = grid_variable
device_variable = create_device_variable()

#################### Constraints ####################
# power_flow_equation(grid_variable)
lindistflow_constraints(grid_variable)
operational_constraints(grid_variable)
nodal_injection_equation(grid_variable, device_variable)

#################### Objective ####################
objective_func = sum(RTP[t] * cp.real(s[f"0_{t}"][0]) for t in range(T))
# Objective
objective = cp.Minimize(objective_func)

problem = cp.Problem(objective, constraints)

#################### Results ####################
result = problem.solve()
print("Optimal value:", result)

#################### Verify ####################
L =np.array([
    [lam["01_0"].value[0,0], 0, 0],
    [0, lam["01_0"].value[1,0], 0],
    [0, 0, lam["01_0"].value[2,0]]
])
a = gamma @ L
print(a)
print(S[f"01_0"].value)
print("####")
print(va["0_0"].value)
print(np.linalg.matrix_rank(a*B))
print(np.linalg.matrix_rank(v[f"0_0"].value))
print(v[f"0_0"].value)
print(va["0_0"].value*B)
