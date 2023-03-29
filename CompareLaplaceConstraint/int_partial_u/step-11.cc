/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2001 - 2021 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth, University of Heidelberg, 2001
 */

/*

- get working with one vector in the nullspace
- test a) use 1 for boundary dofs, 0 else
- test b) use all 1.
- move project() into Nullspace struct
- try out GMRES

- put into ASPECT

 */


#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/types.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <algorithm>
#include <deal.II/numerics/vector_tools_mean_value.h>
#include <iostream>
#include <iomanip>
#include <cmath>

namespace Step11
{
  using namespace dealii;

  template <int dim>
  class LaplaceProblem
  {
  public:
    LaplaceProblem(const unsigned int mapping_degree);
    void run();

  private:
    void setup_system();
    void assemble_and_solve();
    void refine_one_cell();
    void write_high_order_mesh(const unsigned cycle);
    void solve();


    Triangulation<dim> triangulation;
    FE_Q<dim>          fe;
    DoFHandler<dim>    dof_handler;
    MappingQ<dim>      mapping;

    SparsityPattern           sparsity_pattern;
    SparseMatrix<double>      system_matrix;
    AffineConstraints<double> mean_value_constraints;
    AffineConstraints<double> compute_global_constraint;

    Vector<double> solution;
    Vector<double> system_rhs;
    Vector<double> global_constraint_vector;

    TableHandler output_table;
  };



  template <int dim>
  LaplaceProblem<dim>::LaplaceProblem(const unsigned int mapping_degree)
    : fe(1)
    , dof_handler(triangulation)
    , mapping(mapping_degree)
  {
    std::cout << "Using mapping with degree " << mapping_degree << ":"
              << std::endl
              << "============================" << std::endl;
  }

  template <int dim>
  void LaplaceProblem<dim>::setup_system()
  {
    dof_handler.distribute_dofs(fe);
    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    const IndexSet all_dofs = DoFTools::extract_dofs(dof_handler, ComponentMask());
    const IndexSet boundary_dofs = DoFTools::extract_boundary_dofs(dof_handler);
    const types::global_dof_index first_dof = all_dofs.nth_index_in_set(0);
    
    compute_global_constraint.clear();
    
    // constructing null vector.
    if (true){
      const unsigned int gauss_degree = std::max(static_cast<unsigned int>(
                 std::ceil(1. * (mapping.get_degree() + 1) / 2)),
               2U);
      QGauss<dim> quadrature_formula(gauss_degree);
      FEValues<dim, dim> fe_values(mapping, fe, QGauss<dim>(gauss_degree), update_values | update_JxW_values );

      const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
      const unsigned int n_q_points = quadrature_formula.size();

      std::vector<types::global_dof_index> local_dof_index(dofs_per_cell);

      Vector<double> local_constraint_vector(dofs_per_cell);
      global_constraint_vector.reinit(dof_handler.n_dofs());

      for (const auto &cell : dof_handler.active_cell_iterators()){
        fe_values.reinit(cell);
        local_constraint_vector = 0;

        

        for (unsigned int q=0; q<n_q_points; ++q)
          for (unsigned int k = 0; k<dofs_per_cell; ++k)
              local_constraint_vector[k] += fe_values.shape_value(k,q) * fe_values.JxW(q);    
          

        cell->get_dof_indices(local_dof_index);
        compute_global_constraint.distribute_local_to_global(local_constraint_vector, 
                                                            local_dof_index,
                                                            global_constraint_vector);
      }
    }
    compute_global_constraint.close();

      mean_value_constraints.clear();
    if (false){
      mean_value_constraints.add_line(first_dof);  
      for ( types::global_dof_index i : boundary_dofs )
        if (i != first_dof)
          mean_value_constraints.add_entry(first_dof, i, -1.0 * global_constraint_vector[i] / global_constraint_vector[0]);

      // mean_value_constraints.print(std::cout);
    }
      mean_value_constraints.close();

    
    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    mean_value_constraints.condense(dsp);
    // compute_global_constraint.condense(dsp);

    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);
  }

  template <int dim>
  void LaplaceProblem<dim>::write_high_order_mesh(const unsigned cycle)
  {
    DataOut<dim> data_out;

    DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;
    data_out.set_flags(flags);

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");

    data_out.build_patches(mapping,
                           mapping.get_degree(),
                           DataOut<dim>::curved_inner_cells);

    std::ofstream file("solution-c=" + std::to_string(cycle) +
                       ".p=" + std::to_string(mapping.get_degree()) + ".vtu");

    data_out.write_vtu(file);
  }

  template <int dim>
  void LaplaceProblem<dim>::assemble_and_solve()
  {
    const unsigned int gauss_degree =
      std::max(static_cast<unsigned int>(
                 std::ceil(1. * (mapping.get_degree() + 1) / 2)),
               2U);
    MatrixTools::create_laplace_matrix(mapping,
                                       dof_handler,
                                       QGauss<dim>(gauss_degree),
                                       system_matrix);
    VectorTools::create_right_hand_side(mapping,
                                        dof_handler,
                                        QGauss<dim>(gauss_degree),
                                        Functions::ConstantFunction<dim>(-2),
                                        system_rhs);
    Vector<double> tmp(system_rhs.size());
    VectorTools::create_boundary_right_hand_side(
      mapping,
      dof_handler,
      QGauss<dim - 1>(gauss_degree),
      Functions::ConstantFunction<dim>(1),
      tmp);
    system_rhs += tmp;

    mean_value_constraints.condense(system_matrix);
    mean_value_constraints.condense(system_rhs);
    solve();
    mean_value_constraints.distribute(solution);
    
    Vector<float> norm_per_cell(triangulation.n_active_cells());
    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      solution,
                                      Functions::ZeroFunction<dim>(),
                                      norm_per_cell,
                                      QGauss<dim>(gauss_degree + 1),
                                      VectorTools::H1_seminorm);
    const double norm =
      VectorTools::compute_global_error(triangulation,
                                        norm_per_cell,
                                        VectorTools::H1_seminorm);
                              
    double mean_value = VectorTools::compute_mean_value(mapping, dof_handler, QGauss<dim>(gauss_degree), solution, 0 );

    output_table.add_value("cells", triangulation.n_active_cells());
    output_table.add_value("|u|_1", norm);
    output_table.add_value("error",
                           std::fabs(norm - std::sqrt(3.14159265358 / 2)));
    output_table.add_value("MeanValue", mean_value);
  }

  template <class VectorType>
  struct Nullspace
  {
    std::vector<VectorType> basis;
  };
  
  
  template <typename Range, typename Domain, typename Payload, class VectorType>
  LinearOperator<Range, Domain, Payload>
  my_operator(const LinearOperator<Range, Domain, Payload> &op,
				                         Nullspace<VectorType> &nullspace)
  {
      LinearOperator<Range, Domain, Payload> return_op;

      return_op.reinit_range_vector  = op.reinit_range_vector;
      return_op.reinit_domain_vector = op.reinit_domain_vector;

      return_op.vmult = [&](Range &dest, const Domain &src) {
          op.vmult(dest, src);   // dest = Phi(src)

          // Projection.
          for (unsigned int i = 0; i < nullspace.basis.size(); ++i)
            {
              double inner_product = nullspace.basis[i]*dest;
	      dest.add( -1.0*inner_product, nullspace.basis[i]);
            }
      };

      return_op.vmult_add = [&](Range &dest, const Domain &src) {
          std::cout << "before vmult_add" << std::endl;
          op.vmult_add(dest, src);  // dest += Phi(src)
          std::cout << "after vmult_add" << std::endl;
      };

      return_op.Tvmult = [&](Domain &dest, const Range &src) {
          std::cout << "before Tvmult" << std::endl;
          op.Tvmult(dest, src);
          std::cout << "after Tvmult" << std::endl;
      };

      return_op.Tvmult_add = [&](Domain &dest, const Range &src) {
          std::cout << "before Tvmult_add" << std::endl;
          op.Tvmult_add(dest, src);
          std::cout << "after Tvmult_add" << std::endl;
      };

      return return_op;
  }

  template <int dim>
  void LaplaceProblem<dim>::solve()
  {

    using VectorType = Vector<double>;

    SolverControl            solver_control(1000, 1e-12*system_rhs.l2_norm());
    // SolverGMRES<VectorType> solver(solver_control);
    SolverCG<VectorType> solver(solver_control);

    PreconditionJacobi<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.0);


    if (true)
      {

        // Defining Nullspace.
        Nullspace<VectorType> nullspace;
        // VectorType nullvector;

        // This is not a genetic case. This construction is for mean value boundary null space.
        // nullvector.reinit(dof_handler.n_dofs());
        // global_constraint_vector /= global_constraint_vector.l2_norm();
      
        const IndexSet all_dofs = DoFTools::extract_dofs(dof_handler, ComponentMask());
        const IndexSet boundary_dofs = DoFTools::extract_boundary_dofs(dof_handler);

        Vector<double> nullvec(dof_handler.n_dofs());
        
        // nullspace.basis.push_back(global_constraint_vector);
        for ( types::global_dof_index i : boundary_dofs )
          nullvec[i] += global_constraint_vector[i];
            // mean_value_constraints.add_entry(first_dof, i, -1.0 * global_constraint_vector[i] / global_constraint_vector[0]);
        
        nullvec /= nullvec.l2_norm();
        nullspace.basis.push_back(nullvec);

        // original matrix, but projector after preconditioner
        auto matrix_op = //my_operator(linear_operator(system_matrix), nullspace);
          linear_operator(system_matrix);
        auto prec_op = my_operator(linear_operator(preconditioner), nullspace);

        // remove nullspace from RHS
        double r=system_rhs*nullvec;
        system_rhs.add(-1.0*r, nullvec);
        // std::cout << "r=" << r << std::endl;
        solver.solve(matrix_op, solution, system_rhs, prec_op);
        // solution.print(std::cout);
        // solver.solve(matrix_op, solution, system_rhs, PreconditionIdentity());
      }

      // solver.solve(system_matrix, solution, system_rhs, preconditioner);
  }

  template <int dim>
  void LaplaceProblem<dim>::refine_one_cell(){

    auto it = triangulation.active_cell_iterators().begin();
    for (auto &cell : triangulation.active_cell_iterators())
        {
          if (cell == *it){
            cell->set_refine_flag();
          }
        }
 
      triangulation.execute_coarsening_and_refinement();
  }

  template <int dim>
  void LaplaceProblem<dim>::run()
  {
    GridGenerator::hyper_ball(triangulation);

    for (unsigned int cycle = 0; cycle < 6; ++cycle)
      {
        setup_system();
        assemble_and_solve();
        // write_high_order_mesh(cycle);

        triangulation.refine_global();
      }


    // After all the data is generated, write a table of results to the
    // screen:
    output_table.set_precision("|u|_1", 6);
    output_table.set_precision("error", 6);
    output_table.set_precision("MeanValue", 6);
    output_table.write_text(std::cout);
    std::cout << std::endl;
  }
} // namespace Step11



int main()
{
  try
    {
      // dealii::deallog.depth_console(99);
      std::cout.precision(5);

      for (unsigned int mapping_degree = 1; mapping_degree <= 1;
           ++mapping_degree)
        Step11::LaplaceProblem<2>(mapping_degree).run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    };

  return 0;
}
