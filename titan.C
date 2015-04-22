/*
 * titan.C
 *
 *  Created on: Mar 4, 2015
 *      Author: haghakha
 */

// Unsteady Nonlinear
// This example shows how a simple, unsteady, nonlinear system of equations
// can be solved in parallel.  This example introduces the concept of the
// inner nonlinear loop for each timestep, and requires a good deal of
// linear algebra number-crunching at each step.
// C++ include files that we need
#include <iostream>
#include <math.h>
#include <algorithm>
#include <sstream>
// Functions to initialize the library.
#include "libmesh/libmesh.h"
// Basic include files needed for the mesh functionality.
#include "libmesh/mesh.h"
// Include file that defines various mesh generation utilities
#include "libmesh/mesh_generation.h"
// Include file that defines (possibly multiple) systems of equations.
#include "libmesh/equation_systems.h"
// Include files that define a simple steady system
//#include "libmesh/fem_system.h"
#include "libmesh/linear_implicit_system.h"
#include "libmesh/transient_system.h"
// for reading input parameters from file
#include "libmesh/getpot.h"

// Define the DofMap, which handles degree of freedom indexing.
#include "libmesh/dof_map.h"

// Define the Finite Element object.
#include "libmesh/fe.h"

// Define Gauss quadrature rules.
#include "libmesh/quadrature_gauss.h"

// Define useful datatypes for finite element
// matrix and vector components.
#include "libmesh/sparse_matrix.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/dense_matrix.h"
#include "libmesh/dense_vector.h"
#include "libmesh/elem.h"

#include "libmesh/exodusII_io.h"

#include "libmesh/dirichlet_boundaries.h"
#include "libmesh/analytic_function.h"

#include "libmesh/string_to_enum.h"
#include "libmesh/getpot.h"

// Define the PerfLog, a performance logging utility.
// It is useful for timing events in a code and giving
// you an idea where bottlenecks lie.
#include "libmesh/perf_log.h"

#include "libmesh/utility.h"

//#include "libmesh/function_base.h"
//#include "libmesh/libmesh_common.h"
//#include "libmesh/exact_solution.h"

const double GEOFLOW_TINY = 0.0001;

// Bring in everything from the libMesh namespace
using namespace libMesh;

void assemble_sw(EquationSystems& es, const std::string& system_name);

void init_cd(EquationSystems& es, const std::string& system_name);

//std::vector<Number> exact_solution(const Real x, const Real y) {
//
//	//Real h = 0.;
//	std::vector<Number> state(3, 0.0);
//	if ((x * x + y * y) <= .1)
//		state[0] = 1.0;
//	return state;
//}
//
//std::vector<Number> exact_value(const Point& p, const std::string&, const std::string&) {
//	return exact_solution(p(0), p(1));
//}

template<typename T> inline T sign(T t) {
	return (t < -1) ? -1 : 1;
}

Number compute_kap(Number vx_x, Number vy_y, Number bedfric, Number intfric);

int main(int argc, char** argv) {

	LibMeshInit init(argc, argv);

	Mesh mesh(init.comm());
//	this is for loading mesh
//	std::string mesh_file = argv[1];//first input is the binary name
//
//	mesh.read (argv[1]);
	MeshTools::Generation::build_square(mesh, 15, 15, -1., 1., -1., 1., QUAD9);

//	this is for writing mesh
//	mesh.write (argv[2]);

//	we will have only one systems of equations
	EquationSystems es(mesh);

//	this shows how we can set the parameters
	es.parameters.set<Real>("dummy") = 42.;

	// read simulation parameters from file
	GetPot args = GetPot("simulation.data");
	double bed_fric = args("material/friction/bed", 35.0);
	double inter_fric = args("material/friction/internal", 25.0);

	es.parameters.set<Number>("bed_fric") = bed_fric * pi / 180.;
	es.parameters.set<Number>("int_fric") = inter_fric * pi / 180.;

//	the only system of equation we have
//	es.add_system<FEMSystem>("SW");
	TransientLinearImplicitSystem& system = es.add_system<TransientLinearImplicitSystem>("SW");

//	first we just create height
//	unsigned int h_var = es.get_system("SW").add_variable("h", SECOND);
	es.get_system("SW").add_variable("h", FIRST);
//		first momentum
	es.get_system("SW").add_variable("p", FIRST);
//		second momentum
	es.get_system("SW").add_variable("q", FIRST);

	es.get_system("SW").attach_assemble_function(assemble_sw);
	es.get_system("SW").attach_init_function(init_cd);

// initialize es
	es.init();

	mesh.print_info();

	es.print_info();

//	Here added when we make it transient

	PerfLog perf_log("Transient Logger");

	es.parameters.set<unsigned int>("linear solver maximum iterations") = 250;
	es.parameters.set<Real>("linear solver tolerance") = TOLERANCE;

	const Real dt = 0.0001;
	system.time = 0.0;
	const unsigned int n_timesteps = 150;

	const unsigned int n_nonlinear_steps = 20;
	const Real nonlinear_tolerance = 1.e-3;

	es.parameters.set<Real>("dt") = dt;

	AutoPtr<NumericVector<Number> > last_nonlinear_soln(system.solution->clone());

	for (unsigned int t_step = 0; t_step < n_timesteps; ++t_step) {
		std::ostringstream file_name;
		if (t_step == 0) {

			file_name << "out_" << std::setw(3) << std::setfill('0') << std::right << t_step << ".e";

			ExodusII_IO(mesh).write_equation_systems(file_name.str(), es);
		}

		system.time += dt;

		std::cout << "\n\n*** Solving time step " << t_step << ", time = " << system.time << " ***"
		    << std::endl;

		*system.old_local_solution = *system.current_local_solution;

		const Real initial_linear_solver_tol = 1.e-6;
		es.parameters.set<Real>("linear solver tolerance") = initial_linear_solver_tol;

		for (unsigned int l = 0; l < n_nonlinear_steps; ++l) {
			last_nonlinear_soln->zero();
			last_nonlinear_soln->add(*system.solution);

//			perf_log.push("linear solve");
			es.get_system("SW").solve();
//			perf_log.pop("linear solve");

			last_nonlinear_soln->add(-1., *system.solution);

			const Real norm_delta = last_nonlinear_soln->l2_norm();

			const unsigned int n_linear_iterations = system.n_linear_iterations();

			const Real final_linear_residual = system.final_linear_residual();
			std::cout << "Linear solver converged at step: " << n_linear_iterations
			    << ", final residual: " << final_linear_residual
			    << "  Nonlinear convergence: ||u - u_old|| = " << norm_delta << std::endl;

			if ((norm_delta < nonlinear_tolerance)
			    && (system.final_linear_residual() < nonlinear_tolerance)) {
				std::cout << " Nonlinear solver converged at step " << l << std::endl;
				break;
			}

			es.parameters.set<Real>("linear solver tolerance") = std::min(
			    Utility::pow<2>(final_linear_residual), initial_linear_solver_tol);

		} // end nonlinear loop

		const unsigned int write_interval = 20;
		if ((t_step + 1) % write_interval == 0) {
			std::ostringstream file_name;
			file_name << "out_" << std::setw(3) << std::setfill('0') << std::right << t_step + 1 << ".e";

			ExodusII_IO(mesh).write_equation_systems(file_name.str(), es);
		}

	} //end timestep loop.

	return 0;
}

void assemble_sw(EquationSystems& es, const std::string& system_name) {
	// It is a good idea to make sure we are assembling
	// the proper system.
	libmesh_assert_equal_to(system_name, "SW");

	PerfLog perf_log("Matrix Assembly");

	// Get a constant reference to the mesh object.
	const MeshBase& mesh = es.get_mesh();

	// The dimension that we are running
	const unsigned int dim = mesh.mesh_dimension();

	// Get a reference to the Convection-Diffusion system object.
	TransientLinearImplicitSystem & system = es.get_system<TransientLinearImplicitSystem>("SW");

	// Numeric ids corresponding to each variable in the system
	const unsigned int h_var = system.variable_number("h");
	const unsigned int p_var = system.variable_number("p");
	const unsigned int q_var = system.variable_number("q");

	// Get the Finite Element type for "u".  Note this will be
	// the same as the type for "v".
	FEType fe_h_type = system.variable_type(h_var);

	// Build a Finite Element object of the specified type for
	// the velocity variables.
	AutoPtr<FEBase> fe_h(FEBase::build(dim, fe_h_type));
	AutoPtr<FEBase> fe_elem_face(FEBase::build(dim, fe_h_type));
	AutoPtr<FEBase> fe_neighbor_face(FEBase::build(dim, fe_h_type));

	// A Gauss quadrature rule for numerical integration.
	// Let the \p FEType object decide what order rule is appropriate.
	QGauss qrule(dim, fe_h_type.default_quadrature_order());
	QGauss qface(dim - 1, fe_h_type.default_quadrature_order());

	// Tell the finite element objects to use our quadrature rule.
	fe_h->attach_quadrature_rule(&qrule);
	fe_elem_face->attach_quadrature_rule(&qface);
	fe_neighbor_face->attach_quadrature_rule(&qface);

	// Here we define some references to cell-specific data that
	// will be used to assemble the linear system.
	//
	// The element Jacobian * quadrature weight at each integration point.
	const std::vector<Real>& JxW = fe_h->get_JxW();

	// The element shape functions evaluated at the quadrature points.
	const std::vector<std::vector<Real> >& phi = fe_h->get_phi();

	// The element shape function gradients for the velocity
	// variables evaluated at the quadrature points.
	const std::vector<std::vector<RealGradient> >& dphi = fe_h->get_dphi();

	// This part is related to face integrations
	const std::vector<std::vector<Real> >& phi_face = fe_elem_face->get_phi();
	const std::vector<std::vector<RealGradient> >& dphi_face = fe_elem_face->get_dphi();
	const std::vector<Real>& JxW_face = fe_elem_face->get_JxW();
	const std::vector<Point>& qface_normals = fe_elem_face->get_normals();
	const std::vector<Point>& qface_points = fe_elem_face->get_xyz();

	// This part is related to face integrations
	const std::vector<std::vector<Real> >& phi_neighbor_face = fe_neighbor_face->get_phi();
	const std::vector<std::vector<RealGradient> >& dphi_neighbor_face = fe_neighbor_face->get_dphi();

	// A reference to the \p DofMap object for this system.  The \p DofMap
	// object handles the index translation from node and element numbers
	// to degree of freedom numbers.  We will talk more about the \p DofMap
	// in future examples.
	const DofMap & dof_map = system.get_dof_map();

	// Define data structures to contain the element matrix
	// and right-hand-side vector contribution.  Following
	// basic finite element terminology we will denote these
	// "Ke" and "Fe".
	DenseMatrix<Number> Ke;
	DenseVector<Number> Fe;

	DenseSubMatrix<Number> Khh(Ke), Khp(Ke), Khq(Ke), Kph(Ke), Kpp(Ke), Kpq(Ke), Kqh(Ke), Kqp(Ke),
	    Kqq(Ke);

	DenseSubVector<Number> Fh(Fe), Fp(Fe), Fq(Fe);

	// This vector will hold the degree of freedom indices for
	// the element.  These define where in the global system
	// the element degrees of freedom get mapped.
	std::vector<dof_id_type> dof_ind;
	std::vector<dof_id_type> dof_ind_h;
	std::vector<dof_id_type> dof_ind_p;
	std::vector<dof_id_type> dof_ind_q;

	// Find out what the timestep size parameter is from the system, and
	// the value of theta for the theta method.  We use implicit Euler (theta=1)
	// for this simulation even though it is only first-order accurate in time.
	// The reason for this decision is that the second-order Crank-Nicolson
	// method is notoriously oscillatory for problems with discontinuous
	// initial data such as the lid-driven cavity.  Therefore,
	// we sacrifice accuracy in time for stability, but since the solution
	// reaches steady state relatively quickly we can afford to take small
	// timesteps.  If you monitor the initial nonlinear residual for this
	// simulation, you should see that it is monotonically decreasing in time.
	const Real dt = es.parameters.get<Real>("dt");

	const Number intfric = es.parameters.get<Number>("int_fric");

	const Number bedfric = es.parameters.get<Number>("bed_fric");

	// Now we will loop over all the elements in the mesh that
	// live on the local processor. We will compute the element
	// matrix and right-hand-side contribution.  In case users later
	// modify this program to include refinement, we will be safe and
	// will only consider the active elements; hence we use a variant of
	// the \p active_elem_iterator.

	MeshBase::const_element_iterator el = mesh.active_local_elements_begin();
	const MeshBase::const_element_iterator end_el = mesh.active_local_elements_end();

	for (; el != end_el; ++el) {
		// Store a pointer to the element we are currently
		// working on.  This allows for nicer syntax later.
		const Elem* elem = *el;
//		perf_log.push("elem_init");

		// Get the degree of freedom indices for the
		// current element.  These define where in the global
		// matrix and right-hand-side this element will
		// contribute to.
		dof_map.dof_indices(elem, dof_ind);
		dof_map.dof_indices(elem, dof_ind_h, h_var);
		dof_map.dof_indices(elem, dof_ind_p, p_var);
		dof_map.dof_indices(elem, dof_ind_q, q_var);

		const unsigned int n_dofs = dof_ind.size();
		const unsigned int n_h_dofs = dof_ind_h.size();
		const unsigned int n_p_dofs = dof_ind_p.size();
		const unsigned int n_q_dofs = dof_ind_q.size();

		// Compute the element-specific data for the current
		// element.  This involves computing the location of the
		// quadrature points (q_point) and the shape functions
		// (phi, dphi) for the current element.
		fe_h->reinit(elem);

		// Zero the element matrix and right-hand side before
		// summing them.  We use the resize member here because
		// the number of degrees of freedom might have changed from
		// the last element.  Note that this will be the case if the
		// element type is different (i.e. the last element was a
		// triangle, now we are on a quadrilateral).
		Ke.resize(n_dofs, n_dofs);
		Fe.resize(n_dofs);

		// Reposition the submatrices...  The idea is this:
		//
		//         -           -          -  -
		//        | Khh  Khmx  Khmy  |        | Fh  |
		//   Ke = | Kmxh Kmxmx Kmxmy |;  Fe = | Fmx |
		//        | Kmyh Kmymx Kmymy |        | Fmy |
		//         -           -          -  -
		//
		// The \p DenseSubMatrix.repostition () member takes the
		// (row_offset, column_offset, row_size, column_size).
		//
		// Similarly, the \p DenseSubVector.reposition () member
		// takes the (row_offset, row_size)
		Khh.reposition(h_var * n_h_dofs, h_var * n_h_dofs, n_h_dofs, n_h_dofs);
		Khp.reposition(h_var * n_h_dofs, p_var * n_h_dofs, n_h_dofs, n_p_dofs);
		Khq.reposition(h_var * n_h_dofs, q_var * n_h_dofs, n_h_dofs, n_q_dofs);

		Kph.reposition(p_var * n_p_dofs, h_var * n_p_dofs, n_p_dofs, n_h_dofs);
		Kpp.reposition(p_var * n_p_dofs, p_var * n_p_dofs, n_p_dofs, n_p_dofs);
		Kpq.reposition(p_var * n_p_dofs, q_var * n_p_dofs, n_p_dofs, n_q_dofs);

		Kqh.reposition(q_var * n_h_dofs, h_var * n_h_dofs, n_q_dofs, n_h_dofs);
		Kqp.reposition(q_var * n_h_dofs, p_var * n_h_dofs, n_q_dofs, n_p_dofs);
		Kqq.reposition(q_var * n_h_dofs, q_var * n_h_dofs, n_q_dofs, n_q_dofs);

		Fh.reposition(h_var * n_h_dofs, n_h_dofs);
		Fp.reposition(p_var * n_h_dofs, n_p_dofs);
		Fq.reposition(q_var * n_h_dofs, n_q_dofs);

//		perf_log.pop("elem_init");

		// Now we will build the element matrix.
		for (unsigned int qp = 0; qp < qrule.n_points(); qp++) {

			// Values to hold the solution & its gradient at the previous timestep.
			Number h = 0., h_old = 0.;
			Number p = 0., p_old = 0.;
			Number q = 0., q_old = 0.;
			Gradient grad_h, grad_h_old;
			Gradient grad_p, grad_p_old;
			Gradient grad_q, grad_q_old;

			// Compute the velocity & its gradient from the previous timestep
			// and the old Newton iterate.
			for (unsigned int l = 0; l < n_h_dofs; l++) {
				// From the old timestep:
				h_old += phi[l][qp] * system.old_solution(dof_ind_h[l]);
				p_old += phi[l][qp] * system.old_solution(dof_ind_p[l]);
				q_old += phi[l][qp] * system.old_solution(dof_ind_p[l]);
				grad_h_old.add_scaled(dphi[l][qp], system.old_solution(dof_ind_h[l]));
				grad_p_old.add_scaled(dphi[l][qp], system.old_solution(dof_ind_p[l]));
				grad_q_old.add_scaled(dphi[l][qp], system.old_solution(dof_ind_q[l]));

				// From the previous Newton iterate:
				h += phi[l][qp] * system.current_solution(dof_ind_h[l]);
				p += phi[l][qp] * system.current_solution(dof_ind_p[l]);
				q += phi[l][qp] * system.current_solution(dof_ind_q[l]);

				grad_h.add_scaled(dphi[l][qp], system.current_solution(dof_ind_h[l]));
				grad_p.add_scaled(dphi[l][qp], system.current_solution(dof_ind_p[l]));
				grad_q.add_scaled(dphi[l][qp], system.current_solution(dof_ind_q[l]));
			}

			// Definitions for convenience.  It is sometimes simpler to do a
			// dot product if you have the full vector at your disposal.
			const NumberVectorValue H_old(h_old, p_old, q_old);
			const NumberVectorValue H(h, p, q);
			const Number h_x = grad_h_old(0);
			const Number h_y = grad_h_old(1);
			const Number p_x = grad_p_old(0);
			const Number p_y = grad_p_old(1);
			const Number q_x = grad_q_old(0);
			const Number q_y = grad_q_old(1);

			// First, an i-loop over the velocity degrees of freedom.
			// We know that n_u_dofs == n_v_dofs so we can compute contributions
			// for both at the same time.
			for (unsigned int i = 0; i < n_h_dofs; i++) {
				Fh(i) += JxW[qp] * (h_old * phi[i][qp] +    // mass-matrix term
				    dt * p_old * dphi[i][qp](0) + // Flux x term for height term
				    dt * q_old * dphi[i][qp](0)); // Flux y term for height term

				if (h_old > GEOFLOW_TINY) { // actually here hast be based on h not h_old, but here we just want to continue

					const Number vx = p_old / h_old;
					const Number vy = q_old / h_old;
					const Number h_inv = 1 / h_old;
					const Number vel = sqrt(vx * vx + vy * vy);

					if (vel > 0.) {

						const Number gx = 0.; // this has to be corrected later
						const Number gy = 0.; // this has to be corrected later
						const Number gz = -9.8; // this has to be corrected later

						const Number g_x = 0.; // this has to be corrected later
						const Number g_y = 0.; // this has to be corrected later
						//const Number g_z = 0.; // this has to be corrected later

						const Number vx_x = h_inv * (p_x - vx * h_x);
						const Number vx_y = h_inv * (p_y - vx * h_y);

						const Number vy_x = h_inv * (q_x - vy * h_x);
						const Number vy_y = h_inv * (q_y - vy * h_y);

						const Number k_ap = compute_kap(vx_x, vy_y, bedfric, intfric);
						const Number invcurve = 0.; // this has to be corrected later

						// x source terms
						const Number sx1 = gx * h_old; // hydrostatic pressure
						const Number sx2 = h_old * k_ap * sign(vx_y) * g_y * h_y * sin(intfric); // internal friction source term (this part has to be corrected currently I assumed gz is constant)
						const Number sx3 = vx / vel * tan(bedfric) * (gz * h_old + h_old * vx * invcurve);

						// y source terms
						const Number sy1 = gy * h_old; // hydrostatic pressure
						const Number sy2 = h_old * k_ap * sign(vy_x) * g_x * h_x * sin(intfric); // internal friction source term (this part has to be corrected currently I assumed gz is constant)
						const Number sy3 = vy / vel * tan(bedfric) * (gz * h_old + h_old * vy * invcurve);

						Fp(i) += JxW[qp] * (p_old * phi[i][qp] +  // mass-matrix term
						    dt * (p_old * vx + .5 * k_ap * gz * h_old * h_old) * dphi[i][qp](1) + // Flux x term for momentum in x direction
						    dt * p_old * vy * dphi[i][qp](1) + // Flux y term for momentum in x direction
						    dt * (sx1 - sx2 - sx3) * phi[i][qp]);

						Fq(i) += JxW[qp] * (q_old * phi[i][qp] +  // mass-matrix term
						    dt * q_old * vx * dphi[i][qp](1) + // Flux x term for momentum in y direction
						    dt * (q_old * vy + .5 * k_ap * gz * h_old * h_old) * dphi[i][qp](1) + // Flux y term for momentum in y direction
						    dt * (sy1 - sy2 - sy3) * phi[i][qp]);

						if (isnan(Fh(i)) || isnan(Fp(i)) || isnan(Fq(i)))
							std::cout << "here something is wrong  " << Fh(i) << "  , " << Fp(i) << "  ,  "
							    << Fq(i) << std::endl;
					}

				} else {

					Fp(i) += JxW[qp] * p_old * phi[i][qp];

					Fq(i) += JxW[qp] * q_old * phi[i][qp];

				}

				// Note that the Fp block is identically zero unless we are using
				// some kind of artificial compressibility scheme...

				// Matrix contributions for the uu and vv couplings.
				for (unsigned int j = 0; j < n_h_dofs; j++) {
					Khh(i, j) += JxW[qp] * phi[i][qp] * phi[j][qp];	// mass matrix term
					Khp(i, j) += 0;
					Khq(i, j) += 0;

					Kph(i, j) += 0;
					Kpp(i, j) += JxW[qp] * phi[i][qp] * phi[j][qp]; // mass matrix term;
					Kpq(i, j) += 0;

					Kqh(i, j) += 0;
					Kqp(i, j) += 0;
					Kqq(i, j) += JxW[qp] * phi[i][qp] * phi[j][qp]; // mass matrix term
				}
			}

		} // end of the quadrature point qp-loop

		// At this point the interior element integration has
		// been completed.  However, we have not yet addressed
		// boundary conditions.  For this example we will only
		// consider simple Dirichlet boundary conditions imposed
		// via the penalty method. The penalty method used here
		// is equivalent (for Lagrange basis functions) to lumping
		// the matrix resulting from the L2 projection penalty
		// approach introduced in example 3.
//		{
//			// The penalty value.  \f$ \frac{1}{\epsilon} \f$
//			const Real penalty = 1.e10;
//
//			// The following loops over the sides of the element.
//			// If the element has no neighbor on a side then that
//			// side MUST live on a boundary of the domain.
//			for (unsigned int s = 0; s < elem->n_sides(); s++)
//				if (elem->neighbor(s) == NULL) {
//					AutoPtr<Elem> side(elem->build_side(s));
//
//					// Loop over the nodes on the side.
//					for (unsigned int ns = 0; ns < side->n_nodes(); ns++) {
//						// Boundary ids are set internally by
//						// build_square().
//						// 0=bottom
//						// 1=right
//						// 2=top
//						// 3=left
//
//						// Set u = 1 on the top boundary, 0 everywhere else
//						const Real h_value =
//								(mesh.get_boundary_info().has_boundary_id(elem,
//										s, 2)) ? 1. : 0.;
//
//						// Set v = 0 everywhere
//						const Real mx_value = 0.;
//
//						// Find the node on the element matching this node on
//						// the side.  That defined where in the element matrix
//						// the boundary condition will be applied.
//						for (unsigned int n = 0; n < elem->n_nodes(); n++)
//							if (elem->node(n) == side->node(ns)) {
//								// Matrix contribution.
//								Khh(n, n) += penalty;
//								Kpp(n, n) += penalty;
//
//								// Right-hand-side contribution.
//								Fh(n) += penalty * h_value;
//								Fp(n) += penalty * mx_value;
//							}
//					} // end face node loop
//				} // end if (elem->neighbor(side) == NULL)
//
//			// Pin the pressure to zero at global node number "pressure_node".
//			// This effectively removes the non-trivial null space of constant
//			// pressure solutions.
//			const bool pin_pressure = true;
//			if (pin_pressure) {
//				const unsigned int pressure_node = 0;
//				const Real my_value = 0.0;
//				for (unsigned int c = 0; c < elem->n_nodes(); c++)
//					if (elem->node(c) == pressure_node) {
//						Kqq(c, c) += penalty;
//						Fq(c) += penalty * my_value;
//					}
//			}
//		} // end boundary condition section

		// If this assembly program were to be used on an adaptive mesh,
		// we would have to apply any hanging node constraint equations.
		dof_map.constrain_element_matrix_and_vector(Ke, Fe, dof_ind);

		// The element matrix and right-hand-side are now built
		// for this element.  Add them to the global matrix and
		// right-hand-side vector.  The \p NumericMatrix::add_matrix()
		// and \p NumericVector::add_vector() members do this for us.
		system.matrix->add_matrix(Ke, dof_ind);
		system.rhs->add_vector(Fe, dof_ind);
	} // end of element loop

	// That's it.
	return;
}
class SWExactSolution {
public:
	SWExactSolution() {
	}

	~SWExactSolution() {
	}

	Real operator()(unsigned int component, Real x, Real y) {
		switch (component) {
			case 0:
				if ((x * x + y * y) < .1)
					return 1.;
				return 0.;

			case 1:
				return 0.;

			case 2:
				return 0.;

			default:
				libmesh_error_msg("Invalid component = " << component);
		}
	}
};
class SolutionFunction: public FunctionBase<Number> {
public:

	SolutionFunction(const unsigned int state_var) :
			_state_var(state_var) {
	}
	~SolutionFunction() {
	}

	virtual Number operator()(const Point&, const Real = 0) {
		libmesh_not_implemented();}

		virtual void operator() (const Point& p,
				const Real,
				DenseVector<Number>& output)
		{
			output.zero();
			const Real x=p(0), y=p(1);
			// libMesh assumes each component of the vector-valued variable is stored
			// contiguously.
			output(_state_var) = soln( 0, x, y );
			output(_state_var+1) = soln( 1, x, y );
			output(_state_var+2) = soln( 2, x, y );
		}

		virtual Number component( unsigned int component_in, const Point& p,
				const Real )
		{
			const Real x=p(0), y=p(1);
			return soln( component_in, x, y );
		}

		virtual AutoPtr<FunctionBase<Number> > clone() const
		{	return AutoPtr<FunctionBase<Number> > (new SolutionFunction(_state_var));}

	private:

		const unsigned int _state_var;
		SWExactSolution soln;
	};

void init_cd(EquationSystems& es, const std::string& system_name) {

	libmesh_assert_equal_to(system_name, "SW");

	// Get a reference to the Convection-Diffusion system object.
	TransientLinearImplicitSystem & system = es.get_system<TransientLinearImplicitSystem>("SW");

	es.parameters.set<Real>("time") = system.time = 0;

	SolutionFunction func(1);

	system.project_solution(&func, NULL);

	return;
}
Number compute_kap(Number vx_x, Number vy_y, Number bedfric, Number intfric) {

	Number kap;

	Number passive = sign(vx_x + vy_y);

	if (passive == 0) {

		kap = 0.;
		return kap;

	} else {

		kap = 2.
		    * (1.
		        - passive
		            * sqrt(fabs(1. - cos(intfric) * cos(intfric) * (1. + tan(bedfric) * tan(bedfric)))))
		    / (cos(intfric) * cos(intfric)) - 1.;
		return kap;
	}

}
