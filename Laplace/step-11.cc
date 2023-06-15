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


#include <boost/container/vector.hpp>
#include <deal.II/lac/sparse_ilu.h>
#include <boost/integer.hpp>
#include <deal.II/base/numbers.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/base/types.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/lac/full_matrix.h>
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
#include <deal.II/grid/grid_refinement.h>

#include <algorithm>
#include <deal.II/numerics/vector_tools_common.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <deal.II/numerics/vector_tools_mean_value.h>
#include <iostream>
#include <iomanip>
#include <cmath>

namespace Step11
{
  using namespace dealii;

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
              // printf("One iteration done! \n");
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
  class Solution : public Function<dim>
  {
  public:
    virtual double value(const Point<dim> & p, const unsigned int component = 0) const override;
  };
 
 
  template <int dim>
  double Solution<dim>::value(const Point<dim> &p,
                              const unsigned int) const
  {
    double return_value = std::sin(numbers::PI * p[0]) * std::cos(numbers::PI * p[1]);
    
    return return_value;
  }
 
 
 
  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    virtual double value(const Point<dim> & p, const unsigned int component = 0) const override;
  };
 
 
  template <int dim>
  double RightHandSide<dim>::value(const Point<dim> &p, const unsigned int) const
  {
    double return_value = 2*numbers::PI * numbers::PI 
                           * std::sin(numbers::PI * p[0]) 
                           * std::cos(numbers::PI * p[1]);

    return return_value;
  }

  template <int dim>
  class LaplaceProblem
  {
  public:
    LaplaceProblem(const unsigned int mapping_degree);
    void run();

  private:
    void setup_system();
    void assemble_and_solve();
    void interpolate_nullspace();
    void construct_neumann_boundary();
    void solve();
    
    void write_high_order_mesh(const unsigned cycle);

    Triangulation<dim> triangulation;
    FE_Q<dim>          fe;
    DoFHandler<dim>    dof_handler;
    MappingQ<dim>      mapping;

    SparsityPattern           sparsity_pattern;
    SparseMatrix<double>      system_matrix;
    AffineConstraints<double> mean_value_constraints;

    Vector<double> solution;
    Vector<double> system_rhs;
    Vector<double> global_constraint;

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
    global_constraint.reinit(dof_handler.n_dofs());


    mean_value_constraints.clear();  
    DoFTools::make_hanging_node_constraints(dof_handler, mean_value_constraints);
    // mean_value_constraints.add_line(0);
    // for (unsigned int i = 1; i < dof_handler.n_dofs(); ++i)
    //   mean_value_constraints.add_entry(0, i, -1 * global_constraint[i]);
    mean_value_constraints.close();

    interpolate_nullspace();

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());

    DoFTools::make_sparsity_pattern(dof_handler, 
                                    dsp, 
                                    mean_value_constraints, 
                                    /*keep_constrained_dofs = */ false);

    DoFTools::make_sparsity_pattern(dof_handler, dsp);

    mean_value_constraints.condense(dsp);

    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);
  }

  template<int dim>
  void LaplaceProblem<dim>::interpolate_nullspace()
  {
    const unsigned int gauss_degree =
      std::max(static_cast<unsigned int>(
                 std::ceil(1. * (mapping.get_degree() + 1) / 2)),
               2U);

    AffineConstraints<double> constructing_constraint;
    constructing_constraint.close();

    QGauss<dim> quadrature_formula(gauss_degree);
    std::vector<types::global_dof_index> local_dof_indices(fe.n_dofs_per_cell());

    FEValues<dim, dim> fe_values(fe, quadrature_formula, update_JxW_values | update_values );
    Vector<double> local_constraint(fe_values.dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators()){
      local_constraint = 0;
      fe_values.reinit(cell);
      for (unsigned int q_point = 0; q_point < fe_values.n_quadrature_points; ++q_point)
        for (unsigned int i = 0; i < fe_values.dofs_per_cell; ++i)
          local_constraint(i) += fe_values.shape_value(i, q_point) * fe_values.JxW(q_point);
          // local_constraint(i) += 1.0;
        
      
    cell->get_dof_indices(local_dof_indices);
    // const unsigned int dofs_per_cell = fe_values.dofs_per_cell;
    // for (unsigned int i = 0; i < dofs_per_cell; ++i)
    //   global_constraint(local_dof_indices[i]) += local_constraint(i);
    constructing_constraint.distribute_local_to_global(local_constraint, local_dof_indices, global_constraint);
    }
    // global_constraint /= global_constraint.l2_norm();
    mean_value_constraints.distribute(global_constraint);
  }


  template <int dim>
  void LaplaceProblem<dim>::construct_neumann_boundary(){

    const unsigned int gauss_degree =
              std::max(static_cast<unsigned int>(
              std::ceil(1. * (mapping.get_degree() + 1) / 2)),
            2U);

    QGauss<dim-1> face_quadrature_formula(gauss_degree+ 1);
    QGauss<dim> quadrature_formula(gauss_degree+ 1);

    FEValues<dim> fe_values(fe, 
                            quadrature_formula,
                            update_values | update_gradients | update_quadrature_points | update_JxW_values);

    FEFaceValues<dim> fe_face_values(fe,
                                     face_quadrature_formula,
                                     update_values  | update_quadrature_points | update_JxW_values);
    
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_face_q_points = face_quadrature_formula.size();
    // const unsigned int n_q_point(quadrature_formula.size());
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell_rhs = 0;
      cell_matrix = 0;
      fe_values.reinit(cell);

      // rhs_func.value_list(fe_values.get_quadrature_points(), rhs_value);

      // if (true)
      for (unsigned int q_point : fe_values.quadrature_point_indices())
      {
        for (unsigned int i : fe_values.dof_indices())
        {
          for (unsigned int j : fe_values.dof_indices())
          {

            cell_matrix(i,j) += fe_values.shape_grad(i,q_point) *
                                fe_values.shape_grad(j,q_point) * 
                                fe_values.JxW(q_point);
          }

          double x = fe_values.quadrature_point(q_point)[0];
          double y = fe_values.quadrature_point(q_point)[1];
          double F_val = 2 * numbers::PI * numbers::PI * std::sin(numbers::PI * x) * std::cos(numbers::PI * y);
          cell_rhs(i) += (fe_values.shape_value(i, q_point) * // phi_i(x_q)
                            F_val *                     // f(x_q)      
                            fe_values.JxW(q_point));              // dx
        }
      }

      // Evaluating boundary values.
      // if(true)
      for (const auto &face : cell->face_iterators())
      {
          if ( face->at_boundary() )
          {
            fe_face_values.reinit(cell,face);
            for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
                { 
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {

                      double x1 = fe_face_values.quadrature_point(q_point)[0];
                      double x2 = fe_face_values.quadrature_point(q_point)[1];
                       
                      double G = 0;

                      if (face->boundary_id() == 1){
                        G = numbers::PI * std::cos(numbers::PI * x1) * std::cos(numbers::PI * x2);
                      } else if (face->boundary_id() == 2){
                        G = numbers::PI * std::sin(numbers::PI * x1) * std::sin(numbers::PI * x2);
                      } else if (face->boundary_id() == 3){
                        G = -1 * numbers::PI * std::cos(numbers::PI * x1) * std::cos(numbers::PI * x2);
                      } else if (face->boundary_id() == 4){
                        G = -1 * numbers::PI * std::sin(numbers::PI * x1) * std::sin(numbers::PI * x2);
                      }

                      cell_rhs(i) += (fe_face_values.shape_value(i, q_point) *  // phi_i(x_q)
                                      G *                                         // g(x_q)
                                      fe_face_values.JxW(q_point));            // dx
                    }
                }
            } 
      }  

      cell->get_dof_indices(local_dof_indices);
      mean_value_constraints.distribute_local_to_global( cell_matrix, cell_rhs, local_dof_indices,system_matrix, system_rhs );
    }
  }

  template <int dim>
  void LaplaceProblem<dim>::assemble_and_solve()
  {
    const unsigned int gauss_degree =
      std::max(static_cast<unsigned int>(
                 std::ceil(1. * (mapping.get_degree() + 1) / 2)),
               2U);

    construct_neumann_boundary();
    
    solve();

    double mean_value = VectorTools::compute_mean_value(mapping, 
                                                        dof_handler, 
                                                        QGauss<dim>(gauss_degree),
                                                        solution, 
                                                        0);


    // solution.add(-mean_value);



    Vector<double> cellwise_error(triangulation.n_active_cells());
    QGauss<dim> quadrature(gauss_degree);
    VectorTools::integrate_difference(dof_handler, 
                                      solution,
                                      Solution<dim>(),
                                      cellwise_error,
                                      quadrature,
                                      VectorTools::L2_norm);

    const double norm =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_error,
                                        VectorTools::L2_norm);

      // printf("Global error is %f \n", norm);


      GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                    cellwise_error,
                                                    0.3,
                                                    0.03);

    output_table.add_value("cells", triangulation.n_active_cells());
    output_table.add_value("error", norm);
    output_table.add_value("MeanValue", mean_value);
    // printf("after table \n");


  }

  template <int dim>
  void LaplaceProblem<dim>::solve()
  {
    using VectorType = Vector<double>;

    SolverControl            solver_control(2500, 1e-12);
    SolverCG<VectorType>     solver(solver_control);

    SparseILU<double> preconditioner;
    preconditioner.initialize(system_matrix,
                              SparseILU<double>::AdditionalData(1e-3));

    // PreconditionSSOR<SparseMatrix<double>> preconditioner;
    // preconditioner.initialize(system_matrix, 1.2);

    if (true){
      // Defining Nullspace.  
      Nullspace<VectorType> nullspace;

      // Vector<double> all_one_constraint(dof_handler.n_dofs());
      // for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i){
      //   all_one_constraint(i) += 1; 
      // }

      global_constraint /= global_constraint.l2_norm();
      // global_constraint.print(std::cout);
      nullspace.basis.push_back(global_constraint);

      // original matrix, but projector after preconditioner
      // auto matrix_op = my_operator(linear_operator(system_matrix), nullspace);
      auto matrix_op = linear_operator(system_matrix);
      auto prec_op = my_operator(linear_operator(preconditioner), nullspace);

      // remove nullspace from RHS
      double r=system_rhs*global_constraint;
      system_rhs.add(-1.0*r, global_constraint);
      solver.solve(system_matrix, solution, system_rhs, preconditioner);
      // solver.solve(matrix_op, solution, system_rhs, prec_op);
      // mean_value_constraints.distribute(solution);

      double solution_inner = solution * global_constraint;
      solution.add(-1 * solution_inner, global_constraint);
    } else {
      
      Vector<double> global_constraint(dof_handler.n_dofs());
      for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i){
        global_constraint(i) += 1; 
      }

      global_constraint /= global_constraint.l2_norm();

      solver.solve(system_matrix, solution, system_rhs, preconditioner);
      
      mean_value_constraints.distribute(solution);
      // global_constraint /= global_constraint.l2_norm();

      double solution_inner = solution * global_constraint;

      solution.add(-1 * solution_inner, global_constraint);

    }
    
  }

  template <int dim>
  void LaplaceProblem<dim>::write_high_order_mesh(const unsigned cycle)
  {
    DataOut<dim> data_out;
  
    DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;
    data_out.set_flags(flags);
  
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(dof_handler, solution, "solution");

    // Vector<double> exact(dof_handler.n_dofs());
    Vector<double> exact;
    exact.reinit(dof_handler.n_dofs());
    exact = 0;
    VectorTools::interpolate(dof_handler, Solution<dim>(), exact);

    data_out.build_patches(mapping,
                          mapping.get_degree(),
                          DataOut<dim>::curved_inner_cells);
  
    std::ofstream file("solution-c=" + std::to_string(cycle) +
                      ".p=" + std::to_string(mapping.get_degree()) + ".vtu");
  
    data_out.write_vtu(file);
  }


  template <int dim>
  void LaplaceProblem<dim>::run()
  {
    GridGenerator::hyper_cube(triangulation, -1.0, 1.0);
    triangulation.refine_global(2);
    for (const auto &cell : triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
        {
          // setup boundary id.
          if (face-> at_boundary()){
            const auto center = face->center();
            if ((std::fabs(center(0) - (1.0)) < 1e-16) ){
              // Left boundary.
              face->set_boundary_id(1);
            } else if ((std::fabs(center(0) - (-1.0)) < 1e-16 )){
              // Right boundary.
              face->set_boundary_id(3); 
            }else if ((std::fabs(center(1) - (-1.0)) < 1e-16 )){
              // top boundary.
              face->set_boundary_id(2); 
            }else if ((std::fabs(center(1) - (1.0)) < 1e-16 )){
              // bottom boundary.
              face->set_boundary_id(4); 
            }
          }
        }


    for (unsigned int cycle = 0; cycle < 7; ++cycle)
      {
        setup_system();
        assemble_and_solve();
        write_high_order_mesh(cycle);

        if (false){
          triangulation.execute_coarsening_and_refinement();
        } else { 
          triangulation.refine_global();
        }

      }

    output_table.set_precision("error", 6);
    output_table.set_precision("MeanValue", 6);
    output_table.write_text(std::cout);
    // std::cout << std::endl;
  }
} 



int main()
{
  try
    {
      dealii::deallog.depth_console(99);
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
