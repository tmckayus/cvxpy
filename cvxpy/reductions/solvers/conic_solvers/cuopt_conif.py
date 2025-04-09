"""
Copyright 2016 Sascha-Dominic Schnug

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np

import cvxpy.settings as s
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers.conic_solvers.conic_solver import (
    ConicSolver,
    dims_to_solver_dict,
)
from cvxpy.constraints import NonPos

class CUOPT(ConicSolver):
    """ An interface to the CBC solver
    """

    # Solver capabilities.
    MIP_CAPABLE = True
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [NonPos]
    MI_SUPPORTED_CONSTRAINTS = SUPPORTED_CONSTRAINTS

    # Map of CBC status to CVXPY status.
    STATUS_MAP_MIP = {'solution': s.OPTIMAL,
                      'relaxation infeasible': s.INFEASIBLE,
                      'problem proven infeasible': s.INFEASIBLE,
                      'relaxation abandoned': s.SOLVER_ERROR,
                      'stopped on user event': s.SOLVER_ERROR,
                      'stopped on nodes': s.OPTIMAL_INACCURATE,
                      'stopped on gap': s.OPTIMAL_INACCURATE,
                      'stopped on time': s.OPTIMAL_INACCURATE,
                      'stopped on solutions': s.OPTIMAL_INACCURATE,
                      'linear relaxation unbounded': s.UNBOUNDED,
                      'unset': s.UNBOUNDED}

    # LP termination reasons
   # NoTermination    = 0,
   # Optimal          = 1,
   # PrimalInfeasible = 2,
   # DualInfeasible   = 3,
   # IterationLimit   = 4,
   # TimeLimit        = 5,
   # PrimalFeasible   = 6,

    STATUS_MAP_LP = {1: s.OPTIMAL,
                     2: s.INFEASIBLE,
                     3: s.UNBOUNDED,
                     4: s.SOLVER_ERROR,
                     5: s.SOLVER_ERROR,
                     0: s.SOLVER_ERROR}

    def name(self):
        """The name of the solver.
        """
        return s.CUOPT

    def import_solver(self) -> None:
        """Imports the solver.
        """
        try:
            from cuopt.linear_programming.data_model import DataModel
            from cuopt.linear_programming import solver
            self.use_service = False
        except Exception:
            self.use_service = True

    def accepts(self, problem) -> bool:
        """Can cuopt solve the problem?
        """
        # TODO check if is matrix stuffed.
        if not problem.objective.args[0].is_affine():
            return False
        for constr in problem.constraints:
            if type(constr) not in CUOPT.SUPPORTED_CONSTRAINTS:
                return False
            for arg in constr.args:
                if not arg.is_affine():
                    return False
        return True

    def apply(self, problem):
        """Returns a new problem and data for inverting the new solution.

        Returns
        -------
        tuple
            (dict of arguments needed for the solver, inverse data)
        """
        data, inv_data = super(CUOPT, self).apply(problem)
        variables = problem.x
        data[s.BOOL_IDX] = [int(t[0]) for t in variables.boolean_idx]
        data[s.INT_IDX] = [int(t[0]) for t in variables.integer_idx]

        return data, inv_data

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """
        status = solution['status']

        if status in s.SOLUTION_PRESENT:
            opt_val = solution['value'] + inverse_data[s.OFFSET]
            primal_vars = {inverse_data[self.VAR_ID]: solution['primal']}
            return Solution(status, opt_val, primal_vars, None, {})
        else:
            return failure_solution(status)

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):

        csr = data[s.A].tocsr()

        dims = dims_to_solver_dict(data[s.DIMS])
        leq_start = dims[s.EQ_DIM]
        leq_end = dims[s.EQ_DIM] + dims[s.LEQ_DIM]

        num_vars = data['c'].shape[0]
        print('Number of variables: ', num_vars)
        print('Number of integer variables: ', len(data[s.BOOL_IDX]) + len(data[s.INT_IDX]))
        print('Number of equality constraints: ', leq_start)
        print('Number of inequality constraints: ', leq_end - leq_start)

         # No boolean vars available in Cbc -> model as int + restrict to [0,1]
        variable_types = np.array(['C'] * num_vars)
        variable_lower_bounds = np.array([float('-inf') for _ in range(num_vars)])
        variable_upper_bounds = np.array([float('inf') for _ in range(num_vars)])
        if data[s.BOOL_IDX] or data[s.INT_IDX]:
            # Mark integer- and binary-vars as "integer"
            variable_types[data[s.BOOL_IDX]] = 'I'
            variable_types[data[s.INT_IDX]] = 'I'
            variable_lower_bounds[data[s.BOOL_IDX]] = 0
            variable_upper_bounds[data[s.BOOL_IDX]] = 1
        
        lower_bounds = np.concatenate([data['b'][0:leq_start], np.array([float('-inf') for _ in range(leq_start, leq_end)])])
        upper_bounds = np.concatenate([data['b'][0:leq_start], data['b'][leq_start:leq_end]])
        
        if self.use_service:
            import os
            
            d = {}
            d["maximize"] = False
            d["csr_constraint_matrix"] = {
                "offsets": csr.indptr.tolist(),
                "indices": csr.indices.tolist(),
                "values": csr.data.tolist()
            }
            d["objective_data"] = {
                "coefficients": data['c'].tolist(),
                "scalability_factor": 1,
                "offset": 0
            }
            d["variable_bounds"] = {
                "upper_bounds": variable_upper_bounds.tolist(),
                "lower_bounds": variable_lower_bounds.tolist()
            }
            d["constraint_bounds"] = {
                "upper_bounds": upper_bounds.tolist(),
                "lower_bounds": lower_bounds.tolist()
            }
            d["variable_types"] = variable_types.tolist()
            from cuopt_sh_client import CuOptServiceSelfHostClient

            ip = os.environ.get("CUOPT_SERVICE_HOST", "localhost")
            port = os.environ.get("CUOPT_SERVICE_IP", 5000)
            cuopt_service_client = CuOptServiceSelfHostClient(
                ip=ip,
                port=port
            )
            res = cuopt_service_client.get_LP_solve(d, response_type='obj')
            cuopt_result = res["response"]["solver_response"]["solution"]          
        else:
            from cuopt.linear_programming.data_model import DataModel
            from cuopt.linear_programming.solver import Solve
            from cuopt.utilities import setup
            setup()
            
            data_model = DataModel()
            data_model.set_csr_constraint_matrix(csr.data, csr.indices, csr.indptr)
            data_model.set_objective_coefficients(data['c'])
            data_model.set_constraint_lower_bounds(lower_bounds)
            data_model.set_constraint_upper_bounds(upper_bounds)

            # Are bounds for integer variables contained in matrix?
            data_model.set_variable_lower_bounds(variable_lower_bounds)
            data_model.set_variable_upper_bounds(variable_upper_bounds)
            data_model.set_variable_types(variable_types)            
            cuopt_result = Solve(data_model)
        
        print('Termination reason: ', cuopt_result.get_termination_reason())

        solution = {}
        if data[s.BOOL_IDX] or data[s.INT_IDX]:
            solution["status"] = self.STATUS_MAP_MIP[cuopt_result.get_termination_reason()]
        else:
            solution["status"] = self.STATUS_MAP_LP[cuopt_result.get_termination_reason()]
        solution["primal"] = cuopt_result.get_primal_solution()
        solution["value"] = cuopt_result.get_primal_objective()

        return solution

