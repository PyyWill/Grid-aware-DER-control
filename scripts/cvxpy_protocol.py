import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import numpy as np
import torch

class opf_problem:
    def __init__(self, 
                 T: int, 
                 N: int, 
                 Y_common: np.ndarray, 
                 adj_matrix: np.ndarray, 
                 node_class: dict, 
                 RTP: np.ndarray, 
                 demand: torch.tensor):
        """
        Initialize an optimal power flow problem instance.

        Parameters:
        - T (int): The number of time periods.
        - N (int): The number of buses.
        - Z_common (np.ndarray): The common impedance matrix. (change)
        - adj_matrix (np.ndarray): The matrix representing the grid topology
        - node_class (dict): A dictionary categorizing each node into different functionalities. (battery, impedance, PV)
        - RTP (np.ndarray): Real-Time Pricing rates for each time period, an array of length T.
        - demand (np.ndarray): The electrical demand for each node across the time periods.

        Returns:
        None. This is a constructor method for initializing the class instance with the given parameters.
        """
        self.T = T
        self.N = N
        self.Y_common = Y_common
        self.adj_matrix = adj_matrix
        self.RTP = RTP
        self.demand = cp.Parameter(demand.shape, value=demand.clone().detach().cpu().numpy())
        self.battery_node_ls = node_class[0]
        self.impedance_node_ls = node_class[1]
        self.PV_node_ls = node_class[2]

        self.gamma = np.array([
            [1, -0.5 + 0.5 * np.sqrt(3) * 1j, -0.5 - 0.5 * np.sqrt(3) * 1j],
            [-0.5 - 0.5 * np.sqrt(3) * 1j, 1, -0.5 + 0.5 * np.sqrt(3) * 1j],
            [-0.5 + 0.5 * np.sqrt(3) * 1j, -0.5 - 0.5 * np.sqrt(3) * 1j, 1],
        ])
        self.B = np.outer(self.gamma[:,0], self.gamma[:,0].conj())
        self.Y = {}
        self.constraints = []
        self._create_impedance_matrix()
        self.grid_variable = self.create_grid_variable()
        self.device_variable = self.create_device_variable()

    def _create_impedance_matrix(self):
        connections = np.where(self.adj_matrix == 1)
        for i, j in zip(*connections):
            self.Y[f"{i}{j}"] = self.Y_common

    def dfs(self, start, visited=None):
        mat = np.triu(self.adj_matrix)
        if visited is None:
            visited = set()
        visited.add(start)
        for neighbor, isConnected in enumerate(mat[start]):
            if isConnected and neighbor not in visited:
                self.dfs(neighbor, visited)
        return visited

    def find_path(self, target, current=0, visited=None, path=None):
        mat = np.triu(self.adj_matrix)
        if visited is None:
            visited = set()
        if path is None:
            path = []
        visited.add(current)
        if current == target:
            return path
        for neighbor in range(len(mat[current])):
            if self.adj_matrix[current][neighbor] == 1 and neighbor not in visited:
                path.append((current, neighbor))
                result = self.find_path(target, neighbor, visited, path)
                if result is not None:
                    return result
                path.pop()
        return None

    def create_grid_variable(self):
        S = {}; V = {}; I = {}
        s = {}; v = {}; i = {}
        lam = {}; va = {}
        connections = np.where(self.adj_matrix == 1)
        for t in range(self.T):
            for n in range(self.N):
                s[f"{n}_{t}"] = cp.Variable((3, 1), complex=True)
                V[f"{n}_{t}"] = cp.Variable((3, 1), complex=True)
                v[f"{n}_{t}"] = cp.Variable((3, 3), complex=True)
                va[f"{n}_{t}"] = cp.Variable(complex=False)
                self.constraints += [v[f"{n}_{t}"] - va[f"{n}_{t}"]*self.B == 0]
                self.constraints += [v[f"{n}_{t}"] >> 0]
            for j, k in zip(*connections):
                S[f"{j}{k}_{t}"] = cp.Variable((3, 3), complex=True)
                I[f"{j}{k}_{t}"] = cp.Variable((3, 1), complex=True)
                i[f"{j}{k}_{t}"] = cp.Variable((3, 3), complex=True)
                lam[f"{j}{k}_{t}"] = cp.Variable((3, 1), complex=True)
                self.constraints += [i[f"{j}{k}_{t}"] >> 0]
        return (S, V, I, s, v, i, lam, va)

    def create_device_variable(self):
        p_battery = {}; p_PV = {}; p_impedance = {}
        for t in range(self.T):
            for n in self.battery_node_ls:
                p_battery[f"{n}_{t}"] = cp.Variable((3, 1))
            for n in self.PV_node_ls:
                p_PV[f"{n}_{t}"] = cp.Variable((3, 1))
            for n in self.impedance_node_ls:
                p_impedance[f"{n}_{t}"] = cp.Variable((3, 1))
        return (p_battery, p_PV, p_impedance)

    def solve(self):
        self.lindistflow_constraints()
        self.operational_constraints()
        self.nodal_injection_equation()
        # self.power_flow_equation()

        objective_func = sum(self.RTP[t] * cp.real(self.grid_variable[3][f"0_{t}"][0]) for t in range(self.T))
        objective = cp.Minimize(objective_func)
        problem = cp.Problem(objective, self.constraints)
        result = problem.solve()
        return self.grid_variable, self.device_variable


    def lindistflow_constraints(self):
        S, V, I, s, v, i, lam, va = self.grid_variable

        # Total power constraint: Power in should equal power out
        for t in range(self.T):
            self.constraints += [sum(s[f"{n}_{t}"] for n in range(self.N)) == 0]

        # Tree structure constraints: Power flows should be consistent through the tree
        for t in range(self.T):
            connections = np.where(self.adj_matrix == 1)
            for j, k in zip(*connections):
                sub_node = self.dfs(k)
                subtree_power = sum(s[f"{n}_{t}"] for n in sub_node)
                self.constraints += [lam[f"{j}{k}_{t}"] + subtree_power == 0]
                self.constraints += [S[f"{j}{k}_{t}"] - cp.matmul(self.gamma, cp.diag(lam[f"{j}{k}_{t}"])) == 0]

        # Voltage - s constraints
        for t in range(self.T):
            for j, k in zip(*connections):
                if j>k: # for Y_common
                    voltage_diff = self.Y[f"{j}{k}"] @ (v[f"{j}_{t}"]-v[f"{k}_{t}"]) @ cp.conj(self.Y[f"{j}{k}"]).T
                    self.constraints += [voltage_diff - cp.conj(S[f"{j}{k}_{t}"]).T @ cp.conj(self.Y[f"{j}{k}"]).T - self.Y[f"{j}{k}"] @ S[f"{j}{k}_{t}"] == 0]  


    def operational_constraints(self):
        S, V, I, s, v, i, lam, va = self.grid_variable
        connections = np.where(self.adj_matrix == 1)

        # v, s constraints
        for t in range(self.T):
            for n in range(self.N):
                self.constraints += [cp.abs(cp.diag(v[f"{n}_{t}"])[i]) <= 360 for i in range(3)]
                self.constraints += [cp.abs(s[f"{n}_{t}"][i]) <= 1000 for i in range(3)]
        # i constraints
        for j, k in zip(*connections):
            self.constraints += [cp.abs(i[f"{j}{k}_{t}"][p]) <= 100 for p in range(3)]

    def nodal_injection_equation(self):
        S, V, I, s, v, i, lam, va = self.grid_variable
        p_battery, p_PV, p_impedance = self.device_variable

        for t in range(self.T):
            for n in self.PV_node_ls:
                # PV power injection constraints
                self.constraints += [p_PV[f"{n}_{t}"][0] - self.demand + cp.real(s[f"{n}_{t}"])[0] == 0]
                # PV generation capacity constraints
                self.constraints += [0 <= p_PV[f"{n}_{t}"], p_PV[f"{n}_{t}"] <= 20]

            for n in self.battery_node_ls:
                # Battery power constraints
                self.constraints += [p_battery[f"{n}_{t}"][0] - self.demand  + cp.real(s[f"{n}_{t}"])[0] == 0]
                # Battery state-of-charge constraints
                self.constraints += [-10 <= p_battery[f"{n}_{t}"], p_battery[f"{n}_{t}"] <= 10]

            for n in self.impedance_node_ls:
                self.constraints += [p_impedance[f"{n}_{t}"][0] - self.demand  + cp.real(s[f"{n}_{t}"])[0] == 0]


    def power_flow_equation(self):
        S, V, I, s, v, i, lam, va = self.grid_variable
        connections = np.where(self.adj_matrix == 1)
        
        for t in range(self.T):
            for j, k in zip(*connections):
                self.constraints += [S[f"{j}{k}_{t}"] == cp.matmul(V[f"{j}_{t}"], cp.conj(I[f"{j}{k}_{t}"]).T)]
            for n in range(self.N):
                self.constraints += [v[f"{n}_{t}"] == cp.matmul(V[f"{n}_{t}"], cp.conj(V[f"{n}_{t}"]).T)]

class opf_problem_optnn(opf_problem):
    def __init__(self, T, N, Y_common, adj_matrix, node_class, RTP, demand):
        super().__init__(T, N, Y_common, adj_matrix, node_class, RTP, demand)

        self.lindistflow_constraints()
        self.operational_constraints()
        self.nodal_injection_equation()
        # self.power_flow_equation()

        objective_func = sum(self.RTP[t] * cp.real(self.grid_variable[3][f"0_{t}"][0]) for t in range(self.T))
        objective = cp.Minimize(objective_func)
        prob = cp.Problem(objective, self.constraints)
        self.output = []
        p_battery, p_PV, p_impedance = self.device_variable
        for t in range(T):
            for n in range(N):
                if n in self.battery_node_ls:
                    self.output.append(p_battery[f"{n}_{t}"])
                if n in self.PV_node_ls:
                    self.output.append(p_PV[f"{n}_{t}"])
                if n in self.impedance_node_ls:
                    self.output.append(p_impedance[f"{n}_{t}"])

        assert prob.is_dpp()
        self.prob = prob
        self.layer = CvxpyLayer(prob, parameters=[self.demand], variables=self.output)

    def torch_loss(self, demand):
        device_variable = self.forward(demand)
        return self.compute_loss(demand, device_variable)
    
    def compute_loss(self, demand, device_variable):
        p_battery = {}; p_PV = {}; p_impedance = {}
        for t in range(self.T):
            for n in range(self.N):
                if n in self.battery_node_ls:
                    p_battery[f"{n}_{t}"] = device_variable[t]
                if n in self.PV_node_ls:
                    p_PV[f"{n}_{t}"] = device_variable[t+1]
                if n in self.impedance_node_ls:
                    p_impedance[f"{n}_{t}"] = device_variable[t+2]

        def cost_t(t):
            cost_battery_node = sum((demand - p_battery[f"{n}_{t}"][0]) for n in self.battery_node_ls)
            cost_PV_node = sum((demand - p_PV[f"{n}_{t}"][0]) for n in self.PV_node_ls)
            cost_impedance_node = sum((demand - p_impedance[f"{n}_{t}"][0]) for n in self.impedance_node_ls)
            loss_t = cost_battery_node + cost_PV_node + cost_impedance_node
            return loss_t
        cost = sum(-5*cost_t(t) for t in range(self.T))

        return cost
    
    def forward(self, demand):
        solution = self.layer(demand)
        return solution

if __name__ == "__main__":
    Y_common = np.array([
        [0.1 + 0.3j, 0.05 + 0.2j, 0.05 + 0.25j],
        [0.05 + 0.2j, 0.1 + 0.4j, 0.05 + 0.15j],
        [0.05 + 0.25j, 0.05 + 0.15j, 0.1 + 0.5j]
    ])
    RTP = np.array([5, 5])
    T = 1
    adj_matrix = np.array([
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 0]
    ])
    N = 3
    node_class = [[1], [], [2]] # battery, impedance, PV
    demands = [200, 20, 30]

    for demand in demands:
        demand = torch.tensor([[demand]], dtype=torch.float32)
        opf = opf_problem_optnn(T, N, Y_common, adj_matrix, node_class, RTP, demand)
        opf.torch_loss(demand)
