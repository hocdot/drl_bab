from pyscipopt import Model, quicksum
import numpy as np
import copy

def create_milp(c, A, b, bounds):
    # build scip model
    model = Model()
    model.hideOutput()
    # create continuous variables for LP relaxation
    n_con, n_var = A.shape
    X = []
    for i in range(n_var):
        lb, ub = bounds[i]
        x = model.addVar(lb=lb, ub=ub, vtype='C')
        X.append(x)
    # add constraints Ax <= b
    for i in range(n_con):
        model.addCons(quicksum(float(A[i, j]) * X[j] for j in range(n_var)) <= float(b[i]))
    # objective maximize c^T x
    model.setObjective(quicksum(float(c[j]) * X[j] for j in range(n_var)), sense='maximize')
    return model

def copy_milp(milp):
    milp.freeTransform()
    new_milp = Model(sourceModel=milp)
    new_milp.hideOutput()
    return new_milp

class Node:

    def __init__(self, milp):
        self.milp = milp
        self.X = [x for x in milp.getVars()]

    @property
    def primal_estimate(self):
        return self.milp.getObjVal()

    @property
    def dual_estimate(self):
        return self.milp.getDualbound()

    @property
    def active(self):
        # return (self.x_opt != np.round(self.x_opt)).astype(np.int32)
        return (~np.isclose(self.x_opt, np.round(self.x_opt), atol=1e-6)).astype(np.int32)

    @property
    def fraction_score(self):
        return np.abs(self.x_opt - np.round(self.x_opt))

    def branch(self, i):
        # init
        children = []
        x_opt = self.x_opt
        # left child
        milp = copy_milp(self.milp)
        X = [x for x in milp.getVars()]
        milp.addCons(X[i] <= np.floor(x_opt[i]))
        children.append(Node(milp))
        # left child
        milp = copy_milp(self.milp)
        X = [x for x in milp.getVars()]
        milp.addCons(X[i] >= np.ceil(x_opt[i]))
        children.append(Node(milp))
        return children

    def bound(self):
        self.milp.optimize()
        sol = self.milp.getBestSol()
        status = self.milp.getStatus()
        if status == 'optimal':
            self.x_opt = np.array([self.milp.getSolVal(sol, x) for x in self.X], dtype=float)
        return status

    def __del__(self):
        self.milp.freeProb()
