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
	es.get_system("SW").add_variable("q", FIRST);

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

//	 A reference to the  DofMap object for this system.  The  DofMap
//	 object handles the index translation from node and element numbers
//	 to degree of freedom numbers.  We will talk more about the  DofMap
//	 in future examples.
	const DofMap& dof_map = system.get_dof_map();

//	 Get a constant reference to the Finite Element type
//	 for the first (and only) variable in the system.
//	 here we have to be careful if we have more variables
	FEType fe_type = dof_map.variable_type(0);

//	Build a Finite Element object of the specified type.  Since the
//	FEBase::build() member dynamically creates memory we will
//	store the object as an AutoPtr<FEBase>.  This can be thought
//	of as a pointer that will clean up after itself.  Introduction Example 4
//	describes some advantages of  AutoPtr's in the context of
//	quadrature rules.
	AutoPtr<FEBase> fe(FEBase::build(dim, fe_type));

	// A 5th order Gauss quadrature rule for numerical integration.
	QGauss qrule(dim, FIFTH);

	// Tell the finite element object to use our quadrature rule.
	fe->attach_quadrature_rule(&qrule);

	// Declare a special finite element object for
	// boundary integration.
	AutoPtr<FEBase> fe_face(FEBase::build(dim, fe_type));

//	Boundary integration requires one quadraure rule,
//	with dimensionality one less than the dimensionality
//	of the element.
	QGauss qface(dim - 1, FIFTH);

//	Tell the finite element object to use our
//	quadrature rule.
	fe_face->attach_quadrature_rule(&qface);

//	Here we define some references to cell-specific data that
//	will be used to assemble the linear system.
//	The element Jacobian * quadrature weight at each integration point.
	const std::vector<Real>& JxW = fe->get_JxW();

//	The physical XY locations of the quadrature points on the element.
//	These might be useful for evaluating spatially varying material
//	properties at the quadrature points.
	const std::vector<Point>& q_point = fe->get_xyz();

//	The element shape functions evaluated at the quadrature points.
	const std::vector<std::vector<Real> >& phi = fe->get_phi();

//	The element shape function gradients evaluated at the quadrature
//	points.
	const std::vector<std::vector<RealGradient> >& dphi = fe->get_dphi();

//	Define data structures to contain the element matrix
//	and right-hand-side vector contribution.  Following
//	basic finite element terminology we will denote these
//	"Ke" and "Fe".  These datatypes are templated on
//	 Number, which allows the same code to work for real
//	or complex numbers.
	DenseMatrix<Number> Ke;
	DenseVector<Number> Fe;

//	This vector will hold the degree of freedom indices for
//	the element.  These define where in the global system
//	the element degrees of freedom get mapped.
	std::vector<dof_id_type> dof_indices;

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

//		Compute the element-specific data for the current
//		element.  This involves computing the location of the
//		quadrature points (q_point) and the shape functions
//		(phi, dphi) for the current element.
		fe->reinit(elem);

//		Zero the element matrix and right-hand side before
//		summing them.  We use the resize member here because
//		the number of degrees of freedom might have changed from
//		the last element.  Note that this will be the case if the
//		element type is different (i.e. the last element was a
//		triangle, now we are on a quadrilateral).

//		The  DenseMatrix::resize() and the  DenseVector::resize()
//		members will automatically zero out the matrix  and vector.
		Ke.resize(dof_indices.size(), dof_indices.size());

		Fe.resize(dof_indices.size());

		perf_log.pop("elem_init");

//		Now loop over the quadrature points.  This handles
//		the numeric integration.
		for (unsigned int qp = 0; qp < qrule.n_points(); qp++) {

//			Now we will build the element matrix.  This involves
//			a double loop to integrate the test funcions (i) against
//			the trial functions (j).
			for (unsigned int i = 0; i < phi.size(); i++)
				for (unsigned int j = 0; j < phi.size(); j++) {
					Ke(i, j) += JxW[qp] * (dphi[i][qp] * dphi[j][qp]);
				}

//			This is the end of the matrix summation loop
//			Now we build the element right-hand-side contribution.
//			This involves a single loop in which we integrate the
//			"forcing function" in the PDE against the test functions.
			{
//				"fxy" is the forcing function for the Poisson equation.
//				In this case we set fxy to be a finite difference
//				Laplacian approximation to the (known) exact solution.
//
//				We will use the second-order accurate FD Laplacian
//				approximation, which in 2D is
//
//				u_xx + u_yy = 1
//
//				Since the value of the forcing function depends only
//				on the location of the quadrature point (q_point[qp])
//				we will compute it here, outside of the i-loop
//
//				fxy is the forcing function for the Poisson equation.
//				In this case we set fxy to be a finite difference
//				Laplacian approximation to the (known) exact solution.
//
//				We will use the second-order accurate FD Laplacian
//				approximation, which in 2D on a structured grid is
//
//				u_xx + u_yy = (u(i-1,j) + u(i+1,j) +
//				               u(i,j-1) + u(i,j+1) +
//				               -4*u(i,j))/h^2
//
//				Since the value of the forcing function depends only
//				on the location of the quadrature point (q_point[qp])
//				we will compute it here, outside of the i-loop
				const Real x = q_point[qp](0);
				const Real y = q_point[qp](1);
				const Real eps = 1.e-3;

				const Real uxx = (exact_solution(x - eps, y)
						+ exact_solution(x + eps, y)
						+ -2. * exact_solution(x, y)) / eps / eps;

				const Real uyy = (exact_solution(x, y - eps)
						+ exact_solution(x, y + eps)
						+ -2. * exact_solution(x, y)) / eps / eps;

				Real fxy = -(uxx + uyy);
				// Add the RHS contribution
				for (unsigned int i = 0; i < phi.size(); i++)
					Fe(i) += JxW[qp] * fxy * phi[i][qp];

			}
		}
//		We have now reached the end of the RHS summation,
//		and the end of quadrature point loop, so
//		the interior element integration has
//		been completed.  However, we have not yet addressed
//		boundary conditions.  For this example we will only
//		consider simple Dirichlet boundary conditions.
//
//		There are several ways Dirichlet boundary conditions
//		can be imposed.  A simple approach, which works for
//		interpolary bases like the standard Lagrange polynomials,
//		is to assign function values to the
//		degrees of freedom living on the domain boundary. This
//		works well for interpolary bases, but is more difficult
//		when non-interpolary (e.g Legendre or Hierarchic) bases
//		are used.
//
//		Dirichlet boundary conditions can also be imposed with a
//		"penalty" method.  In this case essentially the L2 projection
//		of the boundary values are added to the matrix. The
//		projection is multiplied by some large factor so that, in
//		floating point arithmetic, the existing (smaller) entries
//		in the matrix and right-hand-side are effectively ignored.
//
//		This amounts to adding a term of the form (in latex notation)
//
//		\frac{1}{\epsilon} \int_{\delta \Omega} \phi_i \phi_j = \frac{1}{\epsilon} \int_{\delta \Omega} u \phi_i
//
//		where
//
//		\frac{1}{\epsilon} is the penalty parameter, defined such that \epsilon << 1
		{

//			The following loop is over the sides of the element.
//			If the element has no neighbor on a side then that
//			side MUST live on a boundary of the domain.
			for (unsigned int side = 0; side < elem->n_sides(); side++)
				if (elem->neighbor(side) == NULL) {
//					The value of the shape functions at the quadrature
//					points.
					const std::vector<std::vector<Real> >& phi_face =
							fe_face->get_phi();

//					The Jacobian * Quadrature Weight at the quadrature
//					points on the face.
					const std::vector<Real>& JxW_face = fe_face->get_JxW();

//					 The XYZ locations (in physical space) of the
//					 quadrature points on the face.  This is where
//					 we will interpolate the boundary value function.
					const std::vector<Point>& qface_point = fe_face->get_xyz();

//					 Compute the shape function values on the element
//					 face.
					fe_face->reinit(elem, side);

					// Loop over the face quadrature points for integration.
					for (unsigned int qp = 0; qp < qface.n_points(); qp++) {

						// The location on the boundary of the current
						// face quadrature point.
						const Real xf = qface_point[qp](0);
						const Real yf = qface_point[qp](1);

						const Real penalty = 1.e10;

						// The boundary value.
						const Real value = exact_solution(xf, yf);

						// Matrix contribution of the L2 projection.
						for (unsigned int i = 0; i < phi_face.size(); i++)
							for (unsigned int j = 0; j < phi_face.size(); j++)
								Ke(i, j) += JxW_face[qp] * penalty
										* phi_face[i][qp] * phi_face[j][qp];

						// Right-hand-side contribution of the L2 projection.
						for (unsigned int i = 0; i < phi_face.size(); i++)
							Fe(i) += JxW_face[qp] * penalty * value
									* phi_face[i][qp];
					}
				}
		}

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
		dof_map.constrain_element_matrix_and_vector (Ke, Fe, dof_indices);

//		The element matrix and right-hand-side are now built
//		for this element.  Add them to the global matrix and
//		right-hand-side vector.  The  SparseMatrix::add_matrix()
//		and  NumericVector::add_vector() members do this for us.
		system.matrix->add_matrix(Ke, dof_indices);
		system.rhs->add_vector(Fe, dof_indices);
	}

	return;
}

