/*
 * titan.C
 *
 *  Created on: Mar 4, 2015
 *      Author: haghakha
 */

// C++ include files that we need
#include <iostream>
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

// Bring in everything from the libMesh namespace
using namespace libMesh;

void assemble_sw(EquationSystems& es, const std::string& system_name);

Real exact_solution(const Real x, const Real y) {
	static const Real pi = acos(-1.);

	return cos(.5 * pi * x) * sin(.5 * pi * y);
}

void exact_solution_wrapper(DenseVector<Number>& output, const Point& p,
		const Real) {
	output(0) = exact_solution(p(0), p(1));
}

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
//	GetPot args = GetPot("simulation.data");
//	double bed_fric = args("material/friction/bed", 35.0);
//	double inter_fric = args("material/friction/internal", 25.0);

//	the only system of equation we have
//	es.add_system<FEMSystem>("SW");
	LinearImplicitSystem& system = es.add_system<LinearImplicitSystem>("SW");

//	first we just create height
//	unsigned int h_var = es.get_system("SW").add_variable("h", SECOND);
	es.get_system("SW").add_variable("h", SECOND);
//		first momentum
	es.get_system("SW").add_variable("p", SECOND);
//		second momentum
	es.get_system("SW").add_variable("q", SECOND);

	es.get_system("SW").attach_assemble_function(assemble_sw);

// initialize es
	es.init();

	mesh.print_info();

	es.print_info();

//	es.get_system("SW").solve();
	system.solve();

	ExodusII_IO(mesh).write_equation_systems("out_2.e", es);

	return 0;
}

void assemble_sw(EquationSystems& es, const std::string& system_name) {

	libmesh_assert_equal_to(system_name, "SW");

//	 Declare a performance log.  Give it a descriptive
//	 string to identify what part of the code we are
//	 logging, since there may be many PerfLogs in an
//	 application.
	PerfLog perf_log("Matrix Assembly");

	const MeshBase& mesh = es.get_mesh();

	const unsigned int dim = mesh.mesh_dimension();

//	Get a reference to the LinearImplicitSystem we are solving
//	FEMSystem& system = es.get_system<FEMSystem>("SW");
	LinearImplicitSystem& system = es.get_system<LinearImplicitSystem>("SW");

	// Numeric ids corresponding to each variable in the system
	const unsigned int h_var = system.variable_number("h");
	const unsigned int p_var = system.variable_number("p");
	const unsigned int q_var = system.variable_number("q");

//	 A reference to the  DofMap object for this system.  The  DofMap
//	 object handles the index translation from node and element numbers
//	 to degree of freedom numbers.  We will talk more about the  DofMap
//	 in future examples.
	const DofMap& dof_map = system.get_dof_map();

//	 Get a constant reference to the Finite Element type
//	 for the first (and only) variable in the system.
//	 here we have to be careful if we have more variables
	FEType fe_h_type = dof_map.variable_type(h_var);

//	if I want to use different element types, I have to use the following lines
//	FEType fe_p_type = dof_map.variable_type(p_var);
//	FEType fe_q_type = dof_map.variable_type(q_var);

//	Build a Finite Element object of the specified type.  Since the
//	FEBase::build() member dynamically creates memory we will
//	store the object as an AutoPtr<FEBase>.  This can be thought
//	of as a pointer that will clean up after itself.  Introduction Example 4
//	describes some advantages of  AutoPtr's in the context of
//	quadrature rules.
	AutoPtr<FEBase> fe_h(FEBase::build(dim, fe_h_type));

//	if I want to use different element types, I have to use the following lines
//	AutoPtr<FEBase> fe_p(FEBase::build(dim, fe_p_type));
//	AutoPtr<FEBase> fe_q(FEBase::build(dim, fe_q_type));

// A 5th order Gauss quadrature rule for numerical integration.
//	QGauss qrule(dim, FIFTH);
//	Or
	QGauss qrule(dim, fe_h_type.default_quadrature_order());

	// Tell the finite element object to use our quadrature rule.
	fe_h->attach_quadrature_rule(&qrule);

//	for face elements
//	// Declare a special finite element object for
//	// boundary integration.
//	AutoPtr<FEBase> fe_face(FEBase::build(dim, fe_h_type));
//
////	Boundary integration requires one quadraure rule,
////	with dimensionality one less than the dimensionality
////	of the element.
//	QGauss qface(dim - 1, FIFTH);
//
////	Tell the finite element object to use our
////	quadrature rule.
//	fe_face->attach_quadrature_rule(&qface);

//	Here we define some references to cell-specific data that
//	will be used to assemble the linear system.
//	The element Jacobian * quadrature weight at each integration point.
	const std::vector<Real>& JxW = fe_h->get_JxW();

//	The physical XY locations of the quadrature points on the element.
//	These might be useful for evaluating spatially varying material
//	properties at the quadrature points.
	const std::vector<Point>& q_point = fe_h->get_xyz();

//	The element shape functions evaluated at the quadrature points.
	const std::vector<std::vector<Real> >& phi = fe_h->get_phi();

//	The element shape function gradients evaluated at the quadrature
//	points.
	const std::vector<std::vector<RealGradient> >& dphi = fe_h->get_dphi();

//	Define data structures to contain the element matrix
//	and right-hand-side vector contribution.  Following
//	basic finite element terminology we will denote these
//	"Ke" and "Fe".  These datatypes are templated on
//	 Number, which allows the same code to work for real
//	or complex numbers.
	DenseMatrix<Number> Ke;
	DenseVector<Number> Fe;

	DenseSubMatrix<Number> Khh(Ke), Khp(Ke), Khq(Ke), Kph(Ke), Kpp(Ke), Kpq(Ke),
			Kqh(Ke), Kqp(Ke), Kqq(Ke);

	DenseSubVector<Number> Fh(Fe), Fp(Fe), Fq(Fe);

//	This vector will hold the degree of freedom indices for
//	the element.  These define where in the global system
//	the element degrees of freedom get mapped.
	std::vector<dof_id_type> dof_indices;
	std::vector<dof_id_type> dof_indices_h;
	std::vector<dof_id_type> dof_indices_p;
	std::vector<dof_id_type> dof_indices_q;

//	Now we will loop over all the elements in the mesh.
//	We will compute the element matrix and right-hand-side
//	contribution.

//	Element iterators are a nice way to iterate through all the
//	elements, or all the elements that have some property.  The
//	iterator el will iterate from the first to the last element on
//	the local processor.  The iterator end_el tells us when to stop.
//	It is smart to make this one const so that we don't accidentally
//	mess it up!  In case users later modify this program to include
//	refinement, we will be safe and will only consider the active
//	elements; hence we use a variant of the \p active_elem_iterator.
	MeshBase::const_element_iterator el = mesh.active_local_elements_begin();
	const MeshBase::const_element_iterator end_el =
			mesh.active_local_elements_end();

//	Loop over the elements.  Note that  ++el is preferred to
//	el++ since the latter requires an unnecessary temporary
//	object.
	for (; el != end_el; ++el) {

		perf_log.push("elem_init");

		const Elem* elem = *el;

//		Get the degree of freedom indices for the
//		current element.  These define where in the global
//		matrix and right-hand-side this element will
//		contribute to.
		dof_map.dof_indices(elem, dof_indices);
		dof_map.dof_indices(elem, dof_indices_h, h_var);
		dof_map.dof_indices(elem, dof_indices_p, p_var);
		dof_map.dof_indices(elem, dof_indices_q, q_var);

		const unsigned int n_dofs = dof_indices.size();
		const unsigned int n_h_dofs = dof_indices_h.size();
		const unsigned int n_p_dofs = dof_indices_p.size();
		const unsigned int n_q_dofs = dof_indices_q.size();

//		Compute the element-specific data for the current
//		element.  This involves computing the location of the
//		quadrature points (q_point) and the shape functions
//		(phi, dphi) for the current element.
		fe_h->reinit(elem);

//		Zero the element matrix and right-hand side before
//		summing them.  We use the resize member here because
//		the number of degrees of freedom might have changed from
//		the last element.  Note that this will be the case if the
//		element type is different (i.e. the last element was a
//		triangle, now we are on a quadrilateral).

//		The  DenseMatrix::resize() and the  DenseVector::resize()
//		members will automatically zero out the matrix  and vector.
		Ke.resize(n_dofs, n_dofs);

		Fe.resize(n_dofs);

		// Reposition the submatrices...  The idea is this:
		//
		//         -           -          -  -
		//        | Khh Khp Khq |        | Fh |
		//   Ke = | Kph Kpp Kpq |;  Fe = | Fp |
		//        | Kqh Kqp Kqq |        | Fq |
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

		Kqh.reposition(q_var * n_h_dofs, h_var * n_q_dofs, n_q_dofs, n_h_dofs);
		Kqp.reposition(q_var * n_p_dofs, p_var * n_q_dofs, n_q_dofs, n_p_dofs);
		Kqq.reposition(q_var * n_q_dofs, q_var * n_q_dofs, n_q_dofs, n_q_dofs);

		Fh.reposition(h_var * n_h_dofs, n_h_dofs);
		Fp.reposition(p_var * n_h_dofs, n_p_dofs);
		Fq.reposition(q_var * n_h_dofs, n_q_dofs);

		perf_log.pop("elem_init");

//		Now loop over the quadrature points.  This handles
//		the numeric integration.
		for (unsigned int qp = 0; qp < qrule.n_points(); qp++) {
			// Assemble the u-velocity row
			// uu coupling
			for (unsigned int i = 0; i < n_h_dofs; i++)
				for (unsigned int j = 0; j < n_h_dofs; j++)
					Khh(i, j) += JxW[qp] * (dphi[i][qp] * dphi[j][qp]);

			// up coupling
			for (unsigned int i = 0; i < n_h_dofs; i++)
				for (unsigned int j = 0; j < n_p_dofs; j++)
					Khp(i, j) += -JxW[qp] * phi[j][qp] * dphi[i][qp](0);

			// Assemble the v-velocity row
			// vv coupling
			for (unsigned int i = 0; i < n_p_dofs; i++)
				for (unsigned int j = 0; j < n_p_dofs; j++)
					Kpp(i, j) += JxW[qp] * (dphi[i][qp] * dphi[j][qp]);

			// vp coupling
			for (unsigned int i = 0; i < n_p_dofs; i++)
				for (unsigned int j = 0; j < n_q_dofs; j++)
					Kpq(i, j) += -JxW[qp] * phi[j][qp] * dphi[i][qp](1);

			// Assemble the pressure row
			// pu coupling
			for (unsigned int i = 0; i < n_q_dofs; i++)
				for (unsigned int j = 0; j < n_h_dofs; j++)
					Kqh(i, j) += -JxW[qp] * phi[i][qp] * dphi[j][qp](0);

			// pv coupling
			for (unsigned int i = 0; i < n_q_dofs; i++)
				for (unsigned int j = 0; j < n_p_dofs; j++)
					Kqp(i, j) += -JxW[qp] * phi[i][qp] * dphi[j][qp](1);

		} // end of the quadrature point qp-loop

//		// At this point the interior element integration has
		// been completed.  However, we have not yet addressed
		// boundary conditions.  For this example we will only
		// consider simple Dirichlet boundary conditions imposed
		// via the penalty method. The penalty method used here
		// is equivalent (for Lagrange basis functions) to lumping
		// the matrix resulting from the L2 projection penalty
		// approach introduced in example 3.
		{
			// The following loops over the sides of the element.
			// If the element has no neighbor on a side then that
			// side MUST live on a boundary of the domain.
			for (unsigned int s = 0; s < elem->n_sides(); s++)
				if (elem->neighbor(s) == NULL) {
					AutoPtr<Elem> side(elem->build_side(s));

					// Loop over the nodes on the side.
					for (unsigned int ns = 0; ns < side->n_nodes(); ns++) {
						// The location on the boundary of the current node.

						// const Real xf = side->point(ns)(0);
						const Real yf = side->point(ns)(1);

						// The penalty value.  \f$ \frac{1}{\epsilon \f$
						const Real penalty = 1.e10;

						// The boundary values.

						// Set u = 1 on the top boundary, 0 everywhere else
						const Real h_value = (yf > .99) ? 1. : 0.;

						// Set v = 0 everywhere
						const Real p_value = 0.;

						// Find the node on the element matching this node on
						// the side.  That defined where in the element matrix
						// the boundary condition will be applied.
						for (unsigned int n = 0; n < elem->n_nodes(); n++)
							if (elem->node(n) == side->node(ns)) {
								// Matrix contribution.
								Khh(n, n) += penalty;
								Kpp(n, n) += penalty;

								// Right-hand-side contribution.
								Fh(n) += penalty * h_value;
								Fp(n) += penalty * p_value;
							}
					} // end face node loop
				} // end if (elem->neighbor(side) == NULL)
		} // end boundary condition section
//		We have now finished the quadrature point loop,
//		and have therefore applied all the boundary conditions.

//		If this assembly program were to be used on an adaptive mesh,
//		we would have to apply any hanging node constraint equations
//		dof_map.constrain_element_matrix_and_vector(Ke, Fe, dof_indices);

//		intro example4.C
//		If this assembly program were to be used on an adaptive mesh,
//		we would have to apply any hanging node constraint equations
//		Also, note that here we call heterogenously_constrain_element_matrix_and_vector
//		to impose a inhomogeneous Dirichlet boundary conditions.
//		dof_map.heterogenously_constrain_element_matrix_and_vector(Ke, Fe,
//				dof_indices);

//		based on intro example3.C
		dof_map.constrain_element_matrix_and_vector(Ke, Fe, dof_indices);

//		The element matrix and right-hand-side are now built
//		for this element.  Add them to the global matrix and
//		right-hand-side vector.  The  SparseMatrix::add_matrix()
//		and  NumericVector::add_vector() members do this for us.
		system.matrix->add_matrix(Ke, dof_indices);
		system.rhs->add_vector(Fe, dof_indices);
	}

	return;
}

