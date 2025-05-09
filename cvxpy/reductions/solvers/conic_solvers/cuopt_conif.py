"""
Copyright 2025 NVIDIA CORPORATION

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
from cvxpy.constraints import NonPos
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import (
    ConicSolver,
    dims_to_solver_dict,
)


class CUOPT(ConicSolver):
    """ An interface to the cuOpt solver
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

    STATUS_MAP_MIP = {"NoTermination": s.SOLVER_ERROR,
                      "Optimal": s.OPTIMAL,
                      "FeasibleFound": s.USER_LIMIT,
                      "Infeasible": s.INFEASIBLE,
                      "Unbounded": s.UNBOUNDED}

    # LP termination reasons
    # NoTermination    = 0,
    # Optimal          = 1,
    # PrimalInfeasible = 2,
    # DualInfeasible   = 3,
    # IterationLimit   = 4,
    # TimeLimit        = 5,
    # PrimalFeasible   = 6,

    STATUS_MAP_LP = {"NumericalError": s.SOLVER_ERROR,
                     "Optimal": s.OPTIMAL,
                     "PrimalInfeasible": s.INFEASIBLE,
                     "DualInfeasible": s.INFEASIBLE,
                     "IterationLimit": s.USER_LIMIT,
                     "TimeLimit": s.USER_LIMIT,
                     "PrimalFeasible": s.USER_LIMIT,
                     "ConcurrentLimit": s.SOLVER_ERROR}

    def _solver_mode(self, m):
        from cuopt.linear_programming.solver_settings import SolverMode
        solver_modes = {"Stable1": SolverMode.Stable1,
                        "Stable2": SolverMode.Stable2,
                        "Methodical1": SolverMode.Methodical1,
                        "Fast1": SolverMode.Fast1}
        return solver_modes[m]


    def name(self):
        """The name of the solver.
        """
        return s.CUOPT

    def import_solver(self) -> None:
        """Imports the solver.
        """
        try:
            self.local_install = True
        except Exception:
            self.local_install = False

        try:
            self.service_install = True            
        except Exception:
            self.service_install = False
            if not self.local_install:
                raise

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
                    #print('solution[s.EQ_DUAL] ', solution[s.EQ_DUAL])
                    eq_dual = utilities.get_dual_values(
                        solution[s.EQ_DUAL],
                        utilities.extract_dual_value,
                        inverse_data[self.EQ_CONSTR])
                    dual_vars.update(eq_dual)
                if len(inverse_data[self.NEQ_CONSTR]) > 0:
                    #print('leq')
                    #print('solution[s.INEQ_DUAL] ', solution[s.INEQ_DUAL])
                    #print('inverse_data[self.NEQ_CONSTR] ', inverse_data[self.NEQ_CONSTR])
                    leq_dual = utilities.get_dual_values(
                        solution[s.INEQ_DUAL],
                        utilities.extract_dual_value,
                        inverse_data[self.NEQ_CONSTR])
                    dual_vars.update(leq_dual)


            return Solution(status, opt_val, primal_vars, dual_vars, {})
        else:
            return failure_solution(status)

    # Returns a SolverSettings object
    def _get_solver_settings(self, solver_opts, mip, verbose):
        from cuopt.linear_programming.solver_settings import SolverSettings

        def _apply(name, method):
            if name in solver_opts:
                method(solver_opts[name])
        
        ss = SolverSettings()
        ss.set_log_to_console(verbose)

        ss.set_pdlp_solver_mode(self._solver_mode(solver_opts.get("solver_mode", "Stable2")))
        _apply("absolute_primal_tolerance", ss.set_absolute_primal_tolerance)
        _apply("relative_primal_tolerance", ss.set_relative_primal_tolerance)
        
        if mip:
            # mip currently requires a time, set a default.
            # This requirement will be removed soon.
            #ss.set_time_limit(solver_opts.get("time_limit", 1))
            _apply("mip_scaling", ss.set_mip_scaling)
            _apply("mip_heuristics_only", ss.set_mip_heuristics_only)
            _apply("mip_num_cpu_threads", ss.set_mip_num_cpu_threads)

            # mip-only tolerances
            _apply("integrality_tolerance", ss.set_integrality_tolerance)
        else:
            _apply("time_limit", ss.set_time_limit)
            _apply("infeasibility_detection", ss.set_infeasibility_detection)
            _apply("iteration_limit", ss.set_iteration_limit)
            
            # lp-only tolerances
            _apply("optimality_tolerance", ss.set_optimality_tolerance)
            _apply("absolute_dual_tolerance", ss.set_absolute_dual_tolerance)
            _apply("relative_dual_tolerance", ss.set_relative_dual_tolerance)
            _apply("absolute_gap_tolerance", ss.set_absolute_gap_tolerance)
            _apply("relative_gap_tolerance", ss.set_relative_gap_tolerance)
            _apply("primal_infeasible_tolerance", ss.set_primal_infeasible_tolerance)
            _apply("dual_infeasible_tolerance", ss.set_dual_infeasible_tolerance)

        return ss

    # Returns a dictionary
    def _get_solver_config(self, solver_opts, mip, verbose):

        def _apply(name, sc, alias=None):
            if name in solver_opts:
                if alias is None:
                    alias = name
                sc[alias] = solver_opts[name]

        solver_config = {}
        solver_config["solver_mode"] = self._solver_mode(solver_opts.get("solver_mode", "Stable2"))
        solver_config["log_to_console"] = verbose

        t = {}        
        _apply("absolute_primal_tolerance", t, alias="absolute_primal")
        _apply("relative_primal_tolerance", t, alias="relative_primal")        
        
        if mip:
            # mip currently requires a time, set a default.
            # This requirement will be removed soon.            
            #solver_config["time_limit"] = solver_opts.get("time_limit", 1)
            _apply("mip_scaling", solver_config)
            _apply("mip_heuristics_only", solver_config, alias="heuristics_only")
            _apply("mip_num_cpu_threads", solver_config, alias="num_cpu_threads")

            # mip-only tolerances (note "t")
            _apply("integrality_tolerance", t)
        else:
            _apply("time_limit", solver_config)
            _apply("infeasibility_detection", solver_config)
            _apply("iteration_limit", solver_config)
            
            # lp-only tolerances (note "t")
            _apply("optimality_tolerance", t, alias="optimality")
            _apply("absolute_dual_tolerance", t, alias="absolute_dual")
            _apply("relative_dual_tolerance", t, alias="relative_dual")
            _apply("absolute_gap_tolerance", t, alias="absolute_gap")
            _apply("relative_gap_tolerance", t, alias="relative_gap")
            _apply("primal_infeasible_tolerance", t, alias="primal_infeasible")
            _apply("dual_infeasible_tolerance", t, alias="dual_infeasible")            

        solver_config["tolerances"] = t
        return solver_config

    def _get_client(self, solver_opts):
        import requests
        from cuopt_sh_client import CuOptServiceSelfHostClient

        # Do a health check based on the service arguments        
        ip = solver_opts.get("service_host", "localhost")
        port = solver_opts.get("service_port", 5000)
        scheme = solver_opts.get("service_scheme", "http")
        try:
            loc = f"{scheme}://{ip}:{port}"
            requests.get(f"{loc}/coupt/health")
        except Exception:
            print("Error: cuopt service client is installed but cannot "
                  f"connect to the service at {loc}")
            raise
        return CuOptServiceSelfHostClient(ip=ip, port=port)        


    def _extract_variable_bounds(self, CSR, lower_bounds, upper_bounds, variable_types=None):
        """
        Find single-variable constraints and extract their bounds.

        Args:
            CSR: scipy.sparse.csr_matrix
            lower_bounds: array of lower bounds for constraints
            upper_bounds: array of upper bounds for constraints
            variable_types: array of variable types ('C', 'I', or 'B')
        Returns:
            var_lower_bounds: array of lower bounds for variables
            var_upper_bounds: array of upper bounds for variables
        """
        n_variables = CSR.shape[1]

        # Initialize bounds arrays with infinity
        var_lower_bounds = np.full(n_variables, -np.inf)
        var_upper_bounds = np.full(n_variables, np.inf)

        # If we have variable types, set initial bounds for binary variables
        if variable_types is not None:
            binary_vars = variable_types == 'B'
            var_lower_bounds[binary_vars] = 0
            var_upper_bounds[binary_vars] = 1
        
        # Find rows with exactly one non-zero
        row_nnz = np.diff(CSR.indptr)  # number of non-zeros in each row
        single_coef_rows = row_nnz == 1

        if np.any(single_coef_rows):
            # Get the indices where single coefficients exist
            row_indices = np.where(single_coef_rows)[0]

            for row in row_indices:
                # Get the variable index and coefficient
                var_idx = CSR.indices[CSR.indptr[row]]
                coef = CSR.data[CSR.indptr[row]]

                # If coefficient is positive
                if coef > 0:
                    # Update bounds
                    var_upper_bounds[var_idx] = min(var_upper_bounds[var_idx], 
                                                  upper_bounds[row] / coef)
                    var_lower_bounds[var_idx] = max(var_lower_bounds[var_idx], 
                                                  lower_bounds[row] / coef)
                # If coefficient is negative
                elif coef < 0:
                    # Update bounds (note the swap due to negative coefficient)
                    var_upper_bounds[var_idx] = min(var_upper_bounds[var_idx],
                                                  lower_bounds[row] / coef)
                    var_lower_bounds[var_idx] = max(var_lower_bounds[var_idx],
                                                  upper_bounds[row] / coef)

        # Post-process bounds for integer variables
        if variable_types is not None:

            # Check if we have any integer variables
            integer_vars = variable_types == 'I'
            if np.any(integer_vars):
                # Only process finite bounds for integer variables
                finite_lb = np.isfinite(var_lower_bounds[integer_vars])
                finite_ub = np.isfinite(var_upper_bounds[integer_vars])

                # Update only finite bounds for integer variables
                if np.any(finite_lb):
                    var_lower_bounds[integer_vars][finite_lb] = np.ceil(
                        var_lower_bounds[integer_vars][finite_lb])

                if np.any(finite_ub):
                    var_upper_bounds[integer_vars][finite_ub] = np.floor(
                        var_upper_bounds[integer_vars][finite_ub])

            # Convert binary variables to integer type
            variable_types[binary_vars] = 'I'

        return var_lower_bounds, var_upper_bounds, variable_types

    def add_dummy_single_constraint(self, n_variables):
        from scipy import sparse
        # Create sparse CSR with one row, first element = 1
        data = np.ones(1)
        indices = np.zeros(1, dtype=int)
        indptr = np.array([0, 1])

        CSR = sparse.csr_matrix((data, indices, indptr),
                       shape=(1, n_variables))

        # Inequality constraint bounds (-inf < x <= inf)
        lower_bounds = np.array([0])
        upper_bounds = np.array([10])

        return CSR, lower_bounds, upper_bounds

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):

        use_service = solver_opts.get("use_service", False) in [True,"True","true"]      
        if self.local_install ^ self.service_install:
            if self.local_install:
                if use_service:
                    print("Warning: use_service ignored since cuopt service is not available")
                use_service = False
            else:
                if not use_service:
                    print("Warning: use_service ignored since cuopt is not installed locally")
                use_service = True

        csr = data[s.A].tocsr()
        #csr = data[s.A].tocsr(copy=False)

        num_vars = data['c'].shape[0]

        dims = dims_to_solver_dict(data[s.DIMS])
        leq_start = dims[s.EQ_DIM]
        leq_end = dims[s.EQ_DIM] + dims[s.LEQ_DIM]

        # Get constraint bounds
        import pdb
        pdb.set_trace()
        if dims[s.EQ_DIM] == 0 and dims[s.LEQ_DIM] == 0:
            # No constraints in original problem, add dummy constraints
            n_vars = data['c'].shape[0]
            csr, lower_bounds, upper_bounds = self.add_dummy_single_constraint(n_vars)
        else:
            lower_bounds = np.empty(leq_end)
            lower_bounds[:leq_start] = data['b'][:leq_start]
            lower_bounds[leq_start:leq_end] = float('-inf')

            upper_bounds = data['b'][:leq_end].copy()

       # Determine if we need to extract variable bounds from the constraint matrix
        extract_var_bounds =  "variable_bounds" not in solver_opts or not (
                "lower" in solver_opts["variable_bounds"] and
                "upper" in solver_opts["variable_bounds"])
        
        # Set variable types. We will convert B to I after any necessary
        # bounds extraction so that we can ensure bounds [0,1]
        variable_types = np.empty(num_vars, dtype='U1')
        variable_types.fill('C')
        is_mip = data[s.BOOL_IDX] or data[s.INT_IDX]
        if is_mip:
            if extract_var_bounds:
                # These will be set to I in _extract_variable_bounds after processing
                variable_types[data[s.BOOL_IDX]] = 'B'
            else:
                variable_types[data[s.BOOL_IDX]] = 'I'
            variable_types[data[s.INT_IDX]] = 'I'

        if extract_var_bounds:
            (variable_lower_bounds,
             variable_upper_bounds,
             variable_types) = self._extract_variable_bounds(csr,
                                                             lower_bounds,
                                                             upper_bounds,
                                                             variable_types)

        # Now if we have variable bounds in solver_opts, optionally overwrite lower or upper
        if "variable_bounds" in solver_opts:
            vbounds = solver_opts["variable_bounds"]
            if "lower" in vbounds:
                variable_lower_bounds = vbounds["lower"]
            if "upper" in vbounds:
                variable_upper_bounds = vbounds["upper"]
        
        if use_service:
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
            d["solver_config"] = self._get_solver_config(solver_opts, is_mip, verbose)

            cuopt_service_client = self._get_client(solver_opts)

            # In error case the client will raise an exception here
            res = cuopt_service_client.get_LP_solve(
                d, response_type='obj')["response"]["solver_response"]
            cuopt_result = res["solution"]

            # If conversion to an object didn't work, then this means that
            # we got an infeasible response or similar where expected fields were missing.
            # Since we only need a subset of the object, build it here.
            if isinstance(cuopt_result, dict):
                from cuopt.linear_programming.solution import Solution
                if is_mip:
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
                                        termination_status=res["status"])
            
        else:
            from cuopt.linear_programming.data_model import DataModel
            from cuopt.linear_programming.solver import Solve

            data_model = DataModel()
            data_model.set_csr_constraint_matrix(csr.data, csr.indices, csr.indptr)
            data_model.set_objective_coefficients(data['c'])
            data_model.set_constraint_lower_bounds(lower_bounds)
            data_model.set_constraint_upper_bounds(upper_bounds)

            data_model.set_variable_lower_bounds(variable_lower_bounds)
            data_model.set_variable_upper_bounds(variable_upper_bounds)
            data_model.set_variable_types(variable_types)

            ss = self._get_solver_settings(solver_opts, is_mip, verbose)

            import pickle
            if True:
                with open("cuopt.pickle", "wb") as f:
                    pickle.dump(ss, f)
                    pickle.dump(csr.data, f)
                    pickle.dump(csr.indices, f)
                    pickle.dump(csr.indptr, f)
                    pickle.dump(data['c'], f)
                    pickle.dump(lower_bounds, f)
                    pickle.dump(upper_bounds, f)
                    pickle.dump(variable_lower_bounds, f)
                    pickle.dump(variable_upper_bounds, f)
                    pickle.dump(variable_types, f)

            
            cuopt_result = Solve(data_model, ss)


        print('Termination reason: ', cuopt_result.get_termination_reason())
        
        solution = {}
        if is_mip:
            solution["status"] = self.STATUS_MAP_MIP[cuopt_result.get_termination_reason()]
        else:
            # This really ought to be a getter but the service version of this class is missing it
            # So just grab the result.
            d = cuopt_result.dual_solution
            if d is not None:
                solution[s.EQ_DUAL] = -d[0:leq_start]
                solution[s.INEQ_DUAL] = -d[leq_start:leq_end]
            solution["status"] = self.STATUS_MAP_LP[cuopt_result.get_termination_reason()]

        solution["primal"] = cuopt_result.get_primal_solution()
        solution["value"] = cuopt_result.get_primal_objective()
        return solution

