import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum
import numpy as np

model = gp.Model("complex_optimization")
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

Z_common = np.array([
    [0.1 + 0.3j, 0.05 + 0.2j, 0.05 + 0.25j],
    [0.05 + 0.2j, 0.1 + 0.4j, 0.05 + 0.15j],
    [0.05 + 0.25j, 0.05 + 0.15j, 0.1 + 0.5j]
])

Z = {} # should be a list
connections = np.where(adj_matrix == 1)
for i, j in zip(*connections):
    Z[f"{i}{j}"] = Z_common # Z_ij

RTP = np.array([5,5])

lb = -500 # should be a list
ub = 500 # should be a list
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

def find_path(target, current=0, visited=None, path=None):
    global adj_matrix
    adj_matrix = np.triu(adj_matrix)
    if visited is None:
        visited = set()
    if path is None:
        path = []
    visited.add(current)
    if current == target:
        return path
    for neighbor in range(len(adj_matrix[current])):
        if adj_matrix[current][neighbor] == 1 and neighbor not in visited:
            path.append((current, neighbor))
            result = find_path(target, neighbor, visited, path)
            if result is not None:
                return result
            path.pop()
    return None

def create_grid_variable(N):
    S = {}; V = {}; I = {}
    s = {}; v = {}; i = {}
    lam = {}
    for t in range(T):
        for n in range(N):
            s[f"{n}_re_{t}"] = model.addMVar((3,1), lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=f"s_{n}_real")
            s[f"{n}_im_{t}"] = model.addMVar((3,1), lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=f"s_{n}_imag")
            V[f"{n}_re_{t}"] = model.addMVar((3,1), lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=f"V_{n}_real")
            V[f"{n}_im_{t}"] = model.addMVar((3,1), lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=f"V_{n}_imag")
            v[f"{n}_re_{t}"] = model.addMVar((3,3), lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=f"v_{n}_real")
            v[f"{n}_im_{t}"] = model.addMVar((3,3), lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=f"v_{n}_imag")
        for count, (j, k) in enumerate(zip(*connections)):
            S[f"{j}{k}_re_{t}"] = model.addMVar((3,3), lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=f"S_{j}{k}_real")
            S[f"{j}{k}_im_{t}"] = model.addMVar((3,3), lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=f"S_{j}{k}_imag")
            I[f"{j}{k}_re_{t}"] = model.addMVar((3,1), lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=f"I_{j}{k}_real")
            I[f"{j}{k}_im_{t}"] = model.addMVar((3,1), lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=f"I_{j}{k}_imag")
            i[f"{j}{k}_re_{t}"] = model.addMVar((3,3), lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=f"i_{j}{k}_real")
            i[f"{j}{k}_im_{t}"] = model.addMVar((3,3), lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=f"i_{j}{k}_imag")  
            lam[f"{j}{k}_re_{t}"] = model.addMVar((3,1), lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=f"lambda_{j}{k}_real")
            lam[f"{j}{k}_im_{t}"] = model.addMVar((3,1), lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=f"lambda_{j}{k}_imag")     
    return (S, V, I, s, v, i, lam)

def create_device_variable():
    global battery_node_ls, PV_node_ls, impedance_node_ls
    p_battery = {}; p_PV = {}; p_impedance = {}
    for t in range(T):
        for n in battery_node_ls:
            p_battery[f"{n}_{t}"] = model.addMVar((3,1), lb=0.05*lb, ub=0.05*ub, vtype=GRB.CONTINUOUS)
        for n in PV_node_ls:
            p_PV[f"{n}_{t}"] = model.addMVar((3,1), lb=0, ub=ub, vtype=GRB.CONTINUOUS)
        for n in impedance_node_ls:
            p_impedance[f"{n}_{t}"] = model.addMVar((3,1), lb=lb, ub=0, vtype=GRB.CONTINUOUS)
    return (p_battery, p_PV, p_impedance)

def power_flow_equation(grid_variable):
    S, V, I, s, v, i, lam = grid_variable
    global connections

    # squared equation (v-V, i-I, s-S)
    for t in range(T):
        for k in range(N):
            model.addConstr(v[f"{k}_re_{t}"] - V[f"{k}_re_{t}"] @ V[f"{k}_re_{t}"].T - V[f"{k}_im_{t}"] @ V[f"{k}_im_{t}"].T == 0)
            model.addConstr(v[f"{k}_im_{t}"] + V[f"{k}_re_{t}"] @ V[f"{k}_im_{t}"].T - V[f"{k}_im_{t}"] @ V[f"{k}_re_{t}"].T == 0)
        for j, k in zip(*connections):
            model.addConstr(S[f"{j}{k}_re_{t}"] - V[f"{j}_re_{t}"] @ I[f"{j}{k}_re_{t}"].T - V[f"{j}_im_{t}"] @ I[f"{j}{k}_im_{t}"].T == 0)
            model.addConstr(S[f"{j}{k}_im_{t}"] + V[f"{j}_re_{t}"] @ I[f"{j}{k}_im_{t}"].T - V[f"{j}_im_{t}"] @ I[f"{j}{k}_re_{t}"].T == 0)
    return None

def lindistflow_constraints(grid_variable):
    S, V, I, s, v, i, lam = grid_variable
    global connections

    # tree equation
    for t in range(T):
        model.addConstr(quicksum(s[f"{i}_re_{t}"] for i in range(N)) == 0)
        model.addConstr(quicksum(s[f"{i}_im_{t}"] for i in range(N)) == 0)

    # subtree constraints
    for t in range(T):
        for k, j in zip(*connections):
            sub_node = dfs(j)
            model.addConstr(lam[f"{k}{j}_re_{t}"]+quicksum(s[f"{n}_re_{t}"] for n in sub_node) == 0)
            model.addConstr(lam[f"{k}{j}_im_{t}"]+quicksum(s[f"{n}_im_{t}"] for n in sub_node) == 0)
            
            for p in range(3):
                model.addConstr(S[f"{k}{j}_re_{t}"][:,p]-gamma[:,p].real * lam[f"{k}{j}_re_{t}"][p]+gamma[:,p].imag * lam[f"{k}{j}_im_{t}"][p] == 0)
                model.addConstr(S[f"{k}{j}_im_{t}"][:,p]-gamma[:,p].real * lam[f"{k}{j}_im_{t}"][p]-gamma[:,p].imag * lam[f"{k}{j}_re_{t}"][p] == 0)
     
    # unique path constraints
    for t in range(T):
        for n in range(1, N):
            unique_path = find_path(n)
            model.addConstr(v[f"{n}_re_{t}"]-v[f"0_re_{t}"] + quicksum(Z[f"{j}{k}"].real @ S[f"{j}{k}_re_{t}"].T + S[f"{j}{k}_re_{t}"] @ Z[f"{j}{k}"].real.T + Z[f"{j}{k}"].imag @ S[f"{j}{k}_im_{t}"].T + S[f"{j}{k}_im_{t}"] @ Z[f"{j}{k}"].imag.T for (j,k) in unique_path) == 0)
            model.addConstr(v[f"{n}_im_{t}"]-v[f"0_im_{t}"] + quicksum(Z[f"{j}{k}"].imag @ S[f"{j}{k}_re_{t}"].T + S[f"{j}{k}_im_{t}"] @ Z[f"{j}{k}"].real.T - Z[f"{j}{k}"].real @ S[f"{j}{k}_im_{t}"].T - S[f"{j}{k}_re_{t}"] @ Z[f"{j}{k}"].imag.T for (j,k) in unique_path) == 0)
    return None

def operational_constraints(grid_variable):
    S, V, I, s, v, i, lam = grid_variable
    global connections

    # s constraints
    for t in range(T):
        for n in range(N):
            for p in range(3):
                model.addConstr(s[f"{n}_re_{t}"][p]**2 + s[f"{n}_im_{t}"][p] **2 <= 1000**2)
        # v, i constraints
        for j, k in zip(*connections):
            for p in range(3):
                model.addConstr(i[f"{j}{k}_re_{t}"][p, p] <= 500)
                model.addConstr(i[f"{j}{k}_im_{t}"][p, p] <= 500)
                model.addConstr(i[f"{j}{k}_re_{t}"][p, p]**2 + i[f"{j}{k}_im_{t}"][p, p]**2 <= 500**2)
        for n in range(N):
            for p in range(3):
                model.addConstr(v[f"{n}_re_{t}"][p,p] <= 500)
                model.addConstr(v[f"{n}_im_{t}"][p,p] <= 500)
                model.addConstr(v[f"{n}_re_{t}"][p,p]**2 + v[f"{n}_im_{t}"][p,p]**2 <= 500**2)

    return None

def nodal_injection_equation(grid_variable, device_variable):
    S, V, I, s, v, i, lam = grid_variable
    p_battery, p_PV, p_impedance = device_variable
    global battery_node_ls, PV_node_ls, impedance_node_ls

    for n in PV_node_ls:
        for t in range(T):
            model.addConstr(p_PV[f"{n}_{t}"] - 200 + s[f"{n}_re_{t}"] == 0)
            # PV generation uncertainty
    for n in impedance_node_ls:
        for t in range(T):
            model.addConstr(p_impedance[f"{n}_{t}"] - 200 + s[f"{n}_re_{t}"] == 0)
    for n in battery_node_ls:
        model.addConstr(quicksum(p_battery[f"{n}_{t}"] for t in range(T)) <= 100000)  # SOC constraints
        model.addConstr(quicksum(p_battery[f"{n}_{t}"] for t in range(T)) >= -100000)  # SOC constraints
        for t in range(T):
            model.addConstr(p_battery[f"{n}_{t}"] -200 + s[f"{n}_re_{t}"] == 0)

    return None

#################### Variables ####################
grid_variable = create_grid_variable(N)
S, V, I, s, v, i, lam = grid_variable
device_variable = create_device_variable()

#################### Cons ####################
power_flow_equation(grid_variable)
lindistflow_constraints(grid_variable)
operational_constraints(grid_variable)
nodal_injection_equation(grid_variable, device_variable)

#################### Objective ####################
objective = quicksum(RTP[t] * s[f"0_re_{t}"][0] for t in range(T))

#################### Optimizing ####################
model.setObjective(objective, GRB.MINIMIZE)
model.optimize()
# model.write(r".\test.lp")

#################### Verify ####################
L =np.array([
    [lam["01_re_0"].X[0,0]+lam["01_im_0"].X[0,0]*1j, 0, 0],
    [0, lam["01_re_0"].X[1,0]+lam["01_im_0"].X[1,0]*1j, 0],
    [0, 0, lam["01_re_0"].X[2,0]+lam["01_im_0"].X[2,0]*1j]
])
a = gamma @ L
print(a.real)
print(S[f"01_re_0"].X)
print("###")
print(a.imag)
print(S[f"01_im_0"].X)
print("###")
V_0_0 =np.array([V["0_re_0"].X[0,0]+V["0_im_0"].X[0,0]*1j, V["0_re_0"].X[1,0]+V["0_im_0"].X[1,0]*1j, V["0_re_0"].X[2,0]+V["0_im_0"].X[2,0]*1j])
print(V_0_0)
print(V_0_0[0]*gamma[:, 0])
print("###")
print(np.outer(V_0_0, np.conj(V_0_0)))
print(np.array(v[f"0_re_0"].X)+1j*np.array(v[f"0_im_0"].X))
print("rank v: ", np.linalg.matrix_rank(np.outer(V_0_0, np.conj(V_0_0))))

