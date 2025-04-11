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
import os
import cvxpy.settings as s
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
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

    # NoTermination = 0
    # Optimal = 1
    # FeasibleFound = 2
    # Infeasible = 3
    # Unbounded = 4

    STATUS_MAP_MIP = {0: s.SOLVER_ERROR,
                      1: s.OPTIMAL,
                      2: s.USER_LIMIT,
                      3: s.INFEASIBLE,
                      4: s.UNBOUNDED}

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
            return
        except Exception:
            print("cuOpt is not installed locally. Trying the service ...")

        try:
            import cuopt_sh_client
            import requests
        except Exception:
            print("cuopt_sh_client is not installed.")
            raise

        self.cuopt_ip = os.environ.get("CUOPT_SERVICE_HOST", "localhost")
        self.cuopt_port = os.environ.get("CUOPT_SERVICE_IP", 5000)
        try:
            loc = f"http://{self.cuopt_ip}:{self.cuopt_port}"
            res = requests.get(f"{loc}/coupt/health")
        except Exception:
            print(f"The cuOpt service is not running at {loc}")
            raise
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
        inv_data['lp'] = not (data[s.BOOL_IDX] or data[s.INT_IDX])

        return data, inv_data

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """
        status = solution['status']

        if status in s.SOLUTION_PRESENT:
            dual_vars = None
            opt_val = solution['value'] + inverse_data[s.OFFSET]
            primal_vars = {inverse_data[self.VAR_ID]: solution['primal']}
            if s.EQ_DUAL in solution and inverse_data['lp']:
                dual_vars = {}
                if len(inverse_data[self.EQ_CONSTR]) > 0:
                    print('solution[s.EQ_DUAL] ', solution[s.EQ_DUAL])
                    eq_dual = utilities.get_dual_values(
                        solution[s.EQ_DUAL],
                        utilities.extract_dual_value,
                        inverse_data[self.EQ_CONSTR])
                    dual_vars.update(eq_dual)
                if len(inverse_data[self.NEQ_CONSTR]) > 0:
                    print('leq')
                    print('solution[s.INEQ_DUAL] ', solution[s.INEQ_DUAL])
                    print('inverse_data[self.NEQ_CONSTR] ', inverse_data[self.NEQ_CONSTR])
                    leq_dual = utilities.get_dual_values(
                        solution[s.INEQ_DUAL],
                        utilities.extract_dual_value,
                        inverse_data[self.NEQ_CONSTR])
                    dual_vars.update(leq_dual)


            return Solution(status, opt_val, primal_vars, dual_vars, {})
        else:
            return failure_solution(status)

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):

        csr = data[s.A].tocsr()

        dims = dims_to_solver_dict(data[s.DIMS])
        leq_start = dims[s.EQ_DIM]
        leq_end = dims[s.EQ_DIM] + dims[s.LEQ_DIM]

        num_vars = data['c'].shape[0]
        #print('Number of variables: ', num_vars)
        #print('Number of integer variables: ', len(data[s.BOOL_IDX]) + len(data[s.INT_IDX]))
        #print('Number of equality constraints: ', leq_start)
        #print('Number of inequality constraints: ', leq_end - leq_start)

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

        from cuopt.linear_programming.solver_settings import SolverSettings
        ss = SolverSettings()
        ss.set_solver_mode(3)
        ss.set_time_limit(5)
        
        if self.use_service:
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
            d["solver_config"] = {
                "time_limit": ss.get_time_limit(),
                "solver_mode": ss.get_solver_mode()
            }
            
            from cuopt_sh_client import CuOptServiceSelfHostClient
            ip = os.environ.get("CUOPT_SERVICE_HOST", "localhost")
            port = os.environ.get("CUOPT_SERVICE_IP", 5000)
            cuopt_service_client = CuOptServiceSelfHostClient(
                ip=ip,
                port=port
            )

            # In error case the client will raise an exception here
            res = cuopt_service_client.get_LP_solve(d, response_type='obj', solver_config=ss)["response"]["solver_response"]
            cuopt_result = res["solution"]

            # If conversion to an object didn't work, then this means that we got an infeasible response
            # or similar where expected fields were missing. Since we only need a subset of the object,
            # build it here.
            if isinstance(cuopt_result, dict):
                from cuopt.linear_programming.solution import Solution
                if data[s.BOOL_IDX] or data[s.INT_IDX]:
                    pt = 1
                    dual_solution = None
                else:
                    pt = 0
                    dual_solution = cuopt_result.get("dual_solution", None)
                    if dual_solution:
                        dual_solution = np.array(dual_solution)
                        
                primal_solution = cuopt_result.get("primal_solution", None)
                if primal_solution:
                    primal_solution = np.array(primal_solution)
                primal_objective = cuopt_result.get("primal_objective", 0.0)
                                
                cuopt_result = Solution(problem_category=pt,
                                        vars=None,
                                        dual_solution=dual_solution,
                                        primal_solution=primal_solution,
                                        primal_objective=primal_objective,
                                        termination_reason=res["status"])
            
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

            cuopt_result = Solve(data_model, ss)
            
        #print('Termination reason: ', cuopt_result.get_termination_reason())
        
        solution = {}
        if data[s.BOOL_IDX] or data[s.INT_IDX]:
            solution["status"] = self.STATUS_MAP_MIP[cuopt_result.get_termination_reason()]
        else:
            d = cuopt_result.get_dual_solution()
            if d is not None:
                solution[s.EQ_DUAL] = -d[0:leq_start]
                solution[s.INEQ_DUAL] = -d[leq_start:leq_end]
            solution["status"] = self.STATUS_MAP_LP[cuopt_result.get_termination_reason()]

        solution["primal"] = cuopt_result.get_primal_solution()
        solution["value"] = cuopt_result.get_primal_objective()
        return solution

