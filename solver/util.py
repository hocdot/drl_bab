from pyscipopt import Model, quicksum
import numpy as np


def create_model(c, A, b, bounds):
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

def clone_model(lp_model):
    lp_model.freeTransform()
    new_model = Model(sourceModel=lp_model)
    new_model.hideOutput()
    return new_model

class Node:

    def __init__(self, lp_model):
        self.lp_model = lp_model
        self.X = [x for x in lp_model.getVars()]

    @property
    def primal_estimate(self):
        return self.lp_model.getObjVal()

    @property
    def dual_estimate(self):
        return self.lp_model.getDualbound()

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
        lp_model = clone_model(self.lp_model)
        X = [x for x in lp_model.getVars()]
        lp_model.addCons(X[i] <= np.floor(x_opt[i]))
        children.append(Node(lp_model))
        # left child
        lp_model = clone_model(self.lp_model)
        X = [x for x in lp_model.getVars()]
        lp_model.addCons(X[i] >= np.ceil(x_opt[i]))
        children.append(Node(lp_model))
        return children

    def bound(self):
        self.lp_model.optimize()
        sol = self.lp_model.getBestSol()
        status = self.lp_model.getStatus()
        if status == 'optimal':
            self.x_opt = np.array([self.lp_model.getSolVal(sol, x) for x in self.X], dtype=float)
        return status

    def __del__(self):
        self.lp_model.freeProb()
