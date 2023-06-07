/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2000 - 2021 by the deal.II authors
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
 * Author: Wolfgang Bangerth, University of Heidelberg, 2000
 */


// @sect3{Include files}
#include <bits/types/error_t.h>
#include <boost/container/detail/construct_in_place.hpp>
#include <cmath>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/fe/component_mask.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/base/table_handler.h>

#include <deal.II/numerics/vector_tools_boundary.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <fstream>
#include <iostream>

namespace Step8
{
  using namespace dealii;
  const double lambda_scalar = 2.0;          
  const double pi = numbers::PI;
  const double pi2 = numbers::PI * numbers::PI;



  // Nullspace object.
  template <class VectorType>
  class Nullspace
  {
    public: 
    std::vector<VectorType> basis;
    
    void orthogonalize();
    void remove_nullspace(VectorType& rhs) const;

  };

  template <class VectorType>
  void Nullspace<VectorType>::orthogonalize(){

    // basis.size() = n;
    for (unsigned int j = 0; j < basis.size(); ++j)
    {
      for (unsigned k = 0; k < j; ++k)
      {
        const double tmp_jk = (basis[j]*basis[k])/(basis[k]*basis[k]);
        // printf("ortho between %d and %d is  %f \n",j, k, tmp_jk);
        basis[j].add(-1*tmp_jk, basis[k]);
      }

      basis[j] /= basis[j].l2_norm();
    } 
  }

  template<class VectorType>
  void Nullspace<VectorType>::remove_nullspace(VectorType& input_vector) const{
    for (unsigned int n = 0; n < basis.size(); ++n)
    {
      const double tmp_rhs_inner_product = (basis[n]*input_vector)/(basis[n]*basis[n]);
      printf("Removal %f \n", tmp_rhs_inner_product);
      input_vector.add(-1*tmp_rhs_inner_product, basis[n]);
    }
  }

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
              const double inner_product = nullspace.basis[i]*dest;
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
  void right_hand_side(const std::vector<Point<dim>> &points,
                       std::vector<Tensor<1, dim>> &  values)
  {
    AssertDimension(values.size(), points.size());
    Assert(dim >= 2, ExcNotImplemented());

    for (unsigned int pt = 0; pt < points.size(); ++pt){
      const double x1 = points[pt][0];
      const double x2 = points[pt][1];

      values[pt][0] = -1*pi2*std::sin(pi*x1) * 
                             std::sin(pi*x2) +
                       2*pi2*(1/lambda_scalar + 1)* 
                             std::cos(pi*x1)    * 
                             std::sin(pi*x2);

      values[pt][1] = -1* pi*pi*std::cos(pi*x1) *
                        std::cos(pi * x2)     +
                        2*pi*pi*(1/lambda_scalar + 1)    *
                        std::sin(pi*x1)       *
                        std::cos(pi*x2);
    }
  }

template <int dim>
class Rot: public Function<dim>
{
 public:
    Rot()
      : Function<dim>(dim)
    {}

    virtual void vector_value(const Point<dim> &p,
                              Vector<double> &  values) const override
                              {
                                // values(0) =  -(p(1)-0.5);
                                // values(1) =   (p(0)-0.5);
                                values[0] = -(p(1));
                                values[1] =  (p(0));
                              }
};

template <int dim>
class Translate: public Function<dim>
{
 public:
    Translate(unsigned int comp)
      : Function<dim>(dim), component(comp)
    {}

    virtual void vector_value(const Point<dim> &p,
                              Vector<double> &  values) const override
                              {
                                // values = 0.0;
                                values[component] = 1;//(p(component)-0.5);
                              }

  int component;
};

  template <int dim>
  class ExactSolution : public Function<dim>
  {
  public:
    ExactSolution()
      : Function<dim>(dim)
    {}

    virtual void vector_value(const Point<dim> &p,
                              Vector<double> &  values) const override
                              {
                                const double ux = p[0];
                                const double uy = p[1];

                                values[0] = (-1 * std::sin(pi*ux) + 
                                1/lambda_scalar * std::cos(pi*ux)) *
                                                  std::sin(pi*uy) + 4/pi2;

                                values[1] = (-1 * std::cos(pi * ux) + 
                                1/lambda_scalar * std::sin(pi*ux)) *
                                                  std::cos(pi*uy);
                              }
  };
  
  template <int dim>
  class ElasticProblem
  {
  public:
    ElasticProblem();
    void run();

  private:
    void setup_system();
    void assemble_system();
    void solve();
    void refine_grid();
    void output_results(const unsigned int cycle) const;
    void setup_nullspace();
    void interpolate_nullspace();

    void fixing_three_points();
    void print_mean_value();

    Triangulation<dim> triangulation;
    DoFHandler<dim>    dof_handler;

    FESystem<dim> fe;

    AffineConstraints<double> constraints; // constructing matrix.
    // AffineConstraints<double> construct_nullspace_constraint; // constructing null space;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;
    Nullspace<Vector<double>> nullspace;
    TableHandler output_table;

    Vector<double> solution;
    Vector<double> system_rhs;
    Vector<double> global_rotation;
    Vector<double> global_x_translation;
    Vector<double> global_y_translation;

    double mean_value = 0.0;
    double error_u;
  };

  template <int dim>
  ElasticProblem<dim>::ElasticProblem()
    : dof_handler(triangulation)
    , fe(FE_Q<dim>(1), dim)
  {}
  
  template <int dim>
  void ElasticProblem<dim>::setup_system()
  {
    dof_handler.distribute_dofs(fe);
    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    // Fixing three points or using exact boundary condition.
    if (true)
      {
        fixing_three_points();
      }
      //std::cout << "Not fixing any points \n";
    else
      VectorTools::interpolate_boundary_values(dof_handler,
                                                0,
                                                ExactSolution<dim>(),
                                                constraints);
 
 
    constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    constraints,
                                    /*keep_constrained_dofs = */ false);

    setup_nullspace();
    // interpolate_nullspace();

    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);
  }

  template <int dim>
  void ElasticProblem<dim>::fixing_three_points()
  {

    // So far the best pair for lambda = 2; (1)
    // const Point<dim, double> location_1(0.25,0.5);
    // const Point<dim, double> location_2(1.00,0.5);

    // For lambda = 10: (2)
    // const Point<dim, double> location_1(0.25,0.25);
    // const Point<dim, double> location_2(0.75,0.25);

    // Not very ideal, used to compare: (3)
    const Point<dim, double> location_1(0.0,0.0); // fix x
    const Point<dim, double> location_2(1.0,0.0); // fix y
    // const Point<dim, double> location_3(1.0,1.0); // fix y

    // definitely gonna be there. (4)
    // const Point<dim, double> location_1(0.0,0.0);
    // const Point<dim, double> location_2(1.0,1.0);

    // tradition:(5)
    // const Point<dim, double> location_1(0.0, 0.5);
    // const Point<dim, double> location_2(1.0, 0.5);

    ComponentMask x_direction(2, false);
    x_direction.set(0, true);
    ComponentMask y_direction(2, false);
    y_direction.set(1, true);
    MappingQ<dim> mapping(2);

    const auto &fe = dof_handler.get_fe();
    const std::vector<Point<dim - 1>> &unit_support_points = fe.get_unit_face_support_points();
    const Quadrature<dim - 1> quadrature(unit_support_points);
    const unsigned int dofs_per_face = fe.dofs_per_face;
    std::vector<types::global_dof_index> face_dofs(dofs_per_face);
    const IndexSet all_dof = DoFTools::extract_dofs(dof_handler, ComponentMask());

    FEFaceValues<dim, dim> fe_face_values(mapping,
                                      fe,
                                      quadrature,
                                      update_quadrature_points);

    for (const auto &cell : dof_handler.active_cell_iterators())  
        if (true || cell->at_boundary())
          for (unsigned int face_no = 0;
               face_no < GeometryInfo<dim>::faces_per_cell; ++face_no)
            if (true || cell->at_boundary(face_no))
              {
                const typename DoFHandler<dim, dim>::face_iterator face = cell->face(face_no);
                face->get_dof_indices(face_dofs);
                fe_face_values.reinit(cell, face_no);

                // bool found = false;
                for (unsigned int i = 0; i < face_dofs.size(); ++i)
                  {
                    const unsigned int component = fe.face_system_to_component_index(i).first;
                    if (x_direction[component])
                      {
                        const Point<dim> position = fe_face_values.quadrature_point(i);
                        if (position.distance(location_1) < 1e-2*cell->diameter()){
                            // found = true;
                            if (!constraints.is_constrained(face_dofs[i]) &&
                                constraints.can_store_line(face_dofs[i]) ){
                                  constraints.add_line(face_dofs[i]);
                                  // printf("x1 \n");
                                }
                            } 
                            
                      } else if (y_direction[component]){
                        const Point<dim> position = fe_face_values.quadrature_point(i);
                        if (position.distance(location_1) < 1e-2*cell->diameter())
                          {
                            if (!constraints.is_constrained(face_dofs[i]) &&
                                constraints.can_store_line(face_dofs[i]) ){
                                  constraints.add_line(face_dofs[i]);
                                  // printf("y1 \n");
                                }
                          } else if (position.distance(location_2) < 1e-2*cell->diameter())
                          {
                            // found = true;
                            if (!constraints.is_constrained(face_dofs[i]) &&
                                constraints.can_store_line(face_dofs[i]) ){
                                  constraints.add_line(face_dofs[i]);
                                  // printf("y2 \n");}
                                }
                          }
                      }
                      
                  }

      }

  }

  template <int dim>
  void ElasticProblem<dim>::assemble_system()
  {
    system_matrix = 0;
    system_rhs = 0;

    QGauss<dim> quadrature_formula(fe.degree + 1);
    QGauss<dim-1> face_quadrature_formula(fe.degree + 1);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);


    FEFaceValues<dim> fe_face_values(fe,
                                     face_quadrature_formula,
                                     update_values  | update_quadrature_points | update_JxW_values);


    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();
    
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    std::vector<double> lambda_values(n_q_points);
    std::vector<double> mu_values(n_q_points);
    Functions::ConstantFunction<dim> lambda(lambda_scalar), mu(0.5);

    std::vector<Tensor<1, dim>> rhs_values(n_q_points);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_matrix = 0;
        cell_rhs    = 0;

        fe_values.reinit(cell);

        lambda.value_list(fe_values.get_quadrature_points(), lambda_values);
        mu.value_list(fe_values.get_quadrature_points(), mu_values);
        right_hand_side(fe_values.get_quadrature_points(), rhs_values);
       

        for (const unsigned int i : fe_values.dof_indices())
          {
            const unsigned int component_i =
              fe.system_to_component_index(i).first;

            for (const unsigned int j : fe_values.dof_indices())
              {
                const unsigned int component_j =
                  fe.system_to_component_index(j).first;

                for (const unsigned int q_point :
                     fe_values.quadrature_point_indices())
                  {
                    cell_matrix(i, j) +=
                     
                      (                                                  //
                        (fe_values.shape_grad(i, q_point)[component_i] * //
                         fe_values.shape_grad(j, q_point)[component_j] * //
                         lambda_values[q_point])                         //
                        +                                                //
                        (fe_values.shape_grad(i, q_point)[component_j] * //
                         fe_values.shape_grad(j, q_point)[component_i] * //
                         mu_values[q_point])                             //
                        +                                                //
                       
                        ((component_i == component_j) ?        //
                           (fe_values.shape_grad(i, q_point) * //
                            fe_values.shape_grad(j, q_point) * //
                            mu_values[q_point]) :              //
                           0)                                  //
                        ) *                                    //
                      fe_values.JxW(q_point);                  //
                  }
              }
          }

       
        // Right hand side values.
        for (const unsigned int i : fe_values.dof_indices())
          {
            const unsigned int component_i =
              fe.system_to_component_index(i).first;

            for (const unsigned int q_point :
                 fe_values.quadrature_point_indices())
                 {
                  cell_rhs(i) += fe_values.shape_value(i, q_point) *
                                    rhs_values[q_point][component_i] *
                                    fe_values.JxW(q_point);
                 }
          }

        // boundary conditions.
        // if (true)
        for (const auto &face : cell->face_iterators())
          if ((face->at_boundary()) ){
            fe_face_values.reinit(cell,face);
            for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
                { 
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {

                      double x1 = fe_face_values.quadrature_point(q_point)[0];
                      double x2 = fe_face_values.quadrature_point(q_point)[1];
                       
                      const unsigned int component_i = fe.system_to_component_index(i).first;
                      Tensor<1,dim> G;

                      //Traditional.
                      if (face->boundary_id() == 2 || face->boundary_id() == 4)
                      {
                        G[1] = 0;
                        G[0] = (-1*pi/lambda_scalar * std::cos(pi*x1) );
                      } else {
                        G[0] = (pi * std::sin(pi*x2));
                        G[1] = (-1*pi/lambda_scalar * std::cos(pi*x2));
                      }
                      
                        cell_rhs(i) +=
                                (fe_face_values.shape_value(i, q_point) * // phi_i(x_q)
                                  G[component_i] *                                // 
                                  fe_face_values.JxW(q_point));           // dx
                        
                    }
                }
          } 

     
        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(
          cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
      }
  }

  template <int dim>
  void ElasticProblem<dim>::interpolate_nullspace()
  {

    QGauss<dim> quadrature_formula(fe.degree + 1);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | 
                            update_gradients | 
                            update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

    Vector<double> cell_rotation(dofs_per_cell);
    Vector<double> local_x_translation(dofs_per_cell);
    Vector<double> local_y_translation(dofs_per_cell);

    global_rotation.reinit(dof_handler.n_dofs());
    global_x_translation.reinit(dof_handler.n_dofs());
    global_y_translation.reinit(dof_handler.n_dofs());

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_rotation = 0;
        local_x_translation = 0;
        local_y_translation = 0;

        fe_values.reinit(cell);
        // double area = 0;

        for (const unsigned int i : fe_values.dof_indices())
          {
              for (const unsigned int q_point : fe_values.quadrature_point_indices())
                {
                  const unsigned int component_i = fe.system_to_component_index(i).first;

                  double tmp_grad_xy = fe_values.shape_grad_component(i, q_point, 0)[1]* fe_values.JxW(q_point);
                  double tmp_grad_yx = fe_values.shape_grad_component(i, q_point, 1)[0]* fe_values.JxW(q_point);

                  cell_rotation[i] += tmp_grad_xy - tmp_grad_yx;

                  // printf("On dof %d with q_point %d, the gradient value is %f \n", i, q_point, tmp_grad_xy - tmp_grad_yx);

                  if (component_i == 0)
                  {
                    local_x_translation(i) += fe_values.shape_value(i, q_point) * fe_values.JxW(q_point);
                    // local_x_translation(i) += fe_values.shape_value(i, q_point);
                  }
                  else if (component_i == 1)
                  {
                    local_y_translation(i) += fe_values.shape_value(i, q_point) * fe_values.JxW(q_point);
                    // local_y_translation(i) += fe_values.shape_value(i, q_point);
                  }

                  // area += fe_values.JxW(q_point);
                }
          }
            
        cell->get_dof_indices(local_dof_indices);

        for (unsigned int i = 0; i < dofs_per_cell; ++ i)
        {
          global_rotation[local_dof_indices[i]]      += cell_rotation[i];
          global_x_translation[local_dof_indices[i]] += local_x_translation[i];
          global_y_translation[local_dof_indices[i]] += local_y_translation[i];
        }

        // constraints.distribute_local_to_global( cell_rotation,       local_dof_indices, global_rotation);
        // constraints.distribute_local_to_global( local_x_translation, local_dof_indices, global_x_translation);
        // constraints.distribute_local_to_global( local_y_translation, local_dof_indices, global_y_translation);

      }

      // double sum_quat = 0.0;
      // for (int i = 0; i < global_rotation.size(); ++i){
      //     printf("global_rotation is %f \n", global_rotation[i]);
      //     sum_quat += global_rotation(i);
      // }
      // printf("sum of rotation is %f \n", sum_quat);
      // global_rotation.print(std::cout);
      
      global_rotation /= global_rotation.l2_norm();
      global_x_translation /= global_x_translation.l2_norm();
      global_y_translation /= global_y_translation.l2_norm();

      
  }


  template <int dim>
  void ElasticProblem<dim>::setup_nullspace()
  { 
    global_rotation.reinit(dof_handler.n_dofs());
    global_x_translation.reinit(dof_handler.n_dofs());
    global_y_translation.reinit(dof_handler.n_dofs());

    VectorTools::interpolate(dof_handler, Rot<dim>(), global_rotation);
    global_rotation /= global_rotation.l2_norm();

    VectorTools::interpolate(dof_handler, Translate<dim>(0), global_x_translation);
    global_x_translation /= global_x_translation.l2_norm();

    VectorTools::interpolate(dof_handler, Translate<dim>(1), global_y_translation);
    global_y_translation /= global_y_translation.l2_norm();
  }


  template <int dim>
  void ElasticProblem<dim>::print_mean_value(){
     
    QGauss<dim> quadrature_formula(fe.degree + 1);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    // Vector<double> exact(dof_handler.n_dofs());

    // VectorTools::interpolate(dof_handler, 
    //                           ExactSolution<dim>(),
    //                           exact);

    double mean_value_x = VectorTools::compute_mean_value(fe_values.get_mapping() , 
                    dof_handler, fe_values.get_quadrature(), solution, 0 );

    double mean_value_y = VectorTools::compute_mean_value(fe_values.get_mapping() , 
                    dof_handler, fe_values.get_quadrature(), solution, 1 );
    // printf("mean value of x is %f \n", mean_value_x);
    // printf("mean value of y is %f \n", mean_value_y);
    
    mean_value = std::fabs(mean_value_x) + std::fabs(mean_value_y);
    
  }

  template <int dim>
  void ElasticProblem<dim>::solve()
  {
    SolverControl            solver_control(2000, 1e-12);
    SolverCG<Vector<double>> cg(solver_control);
    // SolverGMRES<Vector<double>> cg(solver_control);
    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);


    // Defining Nullspace.
    nullspace.basis.clear();

    // constraints.set_zero(global_rotation);
    // constraints.set_zero(global_x_translation);
    // constraints.set_zero(global_y_translation);

    constraints.distribute(global_rotation);
    constraints.distribute(global_x_translation);
    constraints.distribute(global_y_translation);

    // Adding vector to basis.            
    nullspace.basis.push_back(global_rotation);
    nullspace.basis.push_back(global_x_translation);
    nullspace.basis.push_back(global_y_translation);

    nullspace.orthogonalize();
    
    constraints.set_zero(nullspace.basis[0]);
    constraints.set_zero(nullspace.basis[1]);
    constraints.set_zero(nullspace.basis[2]);

    // constraints.distribute(nullspace.basis[0]);
    // constraints.distribute(nullspace.basis[1]);
    // constraints.distribute(nullspace.basis[2]);

    // printf("Norm of 0 %f \n", nullspace.basis[0].l2_norm());
    // printf("Norm of 1 %f \n", nullspace.basis[1].l2_norm());
    // printf("Norm of 2 %f \n", nullspace.basis[2].l2_norm());

    // printf("between 0 and 1: %f \n", nullspace.basis[0]*nullspace.basis[1]);
    // printf("between 0 and 2: %f \n", nullspace.basis[0]*nullspace.basis[2]);
    // printf("between 1 and 2: %f \n", nullspace.basis[1]*nullspace.basis[2]);

    // printf("Global rotation \n");
    // nullspace.basis[0].print(std::cout);
    // printf("Global x \n");
    // nullspace.basis[1].print(std::cout);
    // printf("Global y \n");
    // nullspace.basis[2].print(std::cout);
    

    // Operator implementation.
    if (false){
        
        // Solving with null space removal
        printf("Remove null space from right hand side: \n");
        nullspace.remove_nullspace(system_rhs);
        auto matrix_op = linear_operator(system_matrix);
        // auto matrix_op = my_operator(linear_operator(system_matrix), nullspace);
        auto prec_op = my_operator(linear_operator(preconditioner), nullspace);
        cg.solve(matrix_op, solution, system_rhs, prec_op);

        // solution.print(std::cout); 
        constraints.distribute(solution);

        //=================================
        // Remove int_u = int_curl = 0.
        //=================================

        interpolate_nullspace();
        // constraints.set_zero(global_rotation);
        // constraints.set_zero(global_x_translation);
        // constraints.set_zero(global_y_translation);
        // printf("Afterward Removal \n");
        // setup_nullspace();
        // constraints.distribute(global_rotation);
        // constraints.distribute(global_x_translation);
        // constraints.distribute(global_y_translation);
        nullspace.basis.clear();      
        nullspace.basis.push_back(global_rotation);
        nullspace.basis.push_back(global_x_translation);
        nullspace.basis.push_back(global_y_translation);
        nullspace.orthogonalize();
        // constraints.distribute(nullspace.basis[0]);
        // constraints.distribute(nullspace.basis[1]);
        // constraints.distribute(nullspace.basis[2]);

        // printf("Remove null space from solution: \n");
        nullspace.remove_nullspace(solution); 

        // Distribute after all removal
        // constraints.distribute(solution);

        // ======================================================
        // Compute mean value of x and y overall and remove it.
        // ======================================================

        // QGauss<dim> quadrature_formula(fe.degree + 1);

        // FEValues<dim> fe_values(fe,
        //                         quadrature_formula,
        //                         update_values | update_gradients |
        //                           update_quadrature_points | update_JxW_values);

        // double mean_value_x = VectorTools::compute_mean_value(fe_values.get_mapping() , dof_handler, fe_values.get_quadrature(), solution, 0);
        // double mean_value_y = VectorTools::compute_mean_value(fe_values.get_mapping() , dof_handler, fe_values.get_quadrature(), solution, 1);
        
        // solution.add(mean_value_x);
        // solution.add(mean_value_y);

        // Vector<double> x_plot;
        // Vector<double> y_plot;
        // x_plot.reinit(dof_handler.n_dofs());
        // y_plot.reinit(dof_handler.n_dofs());

        // for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i){
        //   if (i % 2 == 1)
        //     x_plot(i) += mean_value_x;
        //   else
        //     y_plot(i) += mean_value_y;
        // }

        // x_plot /= x_plot.l2_norm();
        // y_plot /= y_plot.l2_norm();

        // double tmp_sx = solution * x_plot;
        // solution.add(-tmp_sx, x_plot);

        // double tmp_sy = solution * y_plot;
        // solution.add(-tmp_sy, y_plot);

        print_mean_value();

    } 
    else
    {
        // Traditional Solve.
        cg.solve(system_matrix, solution, system_rhs, preconditioner);

        // Post processing.
        nullspace.remove_nullspace(solution); 
        constraints.distribute(solution);

        // QGauss<dim> quadrature_formula(fe.degree + 1);

        // FEValues<dim> fe_values(fe,
        //                         quadrature_formula,
        //                         update_values | update_gradients |
        //                           update_quadrature_points | update_JxW_values);

        // double mean_value_x = VectorTools::compute_mean_value(fe_values.get_mapping() , dof_handler, fe_values.get_quadrature(), solution, 0);
        // double mean_value_y = VectorTools::compute_mean_value(fe_values.get_mapping() , dof_handler, fe_values.get_quadrature(), solution, 1);
        
        // constraints.set_zero(global_x_translation);
        // constraints.set_zero(global_y_translation);
        // solution.add(-mean_value_x, global_x_translation);
        // solution.add(-0.35, global_y_translation);

        // Vector<double> x_plot;
        // Vector<double> y_plot;
        // x_plot.reinit(dof_handler.n_dofs());
        // y_plot.reinit(dof_handler.n_dofs());

        // for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i){
        //   if (i % 2 == 1)
        //     x_plot(i) += std::fabs(mean_value_x);
        //   else
        //     y_plot(i) += std::fabs(mean_value_y);
        // }

        // x_plot /= x_plot.l2_norm();
        // y_plot /= y_plot.l2_norm();

        // double tmp_sx = solution * x_plot;
        // solution.add(-tmp_sx, x_plot);

        // double tmp_sy = solution * y_plot;
        // solution.add(-tmp_sy, y_plot);

        // interpolate_nullspace();
        // constraints.distribute(global_rotation);
        // constraints.distribute(global_x_translation);
        // constraints.distribute(global_y_translation);
        // constraints.set_zero(global_rotation);
        // constraints.set_zero(global_x_translation);
        // constraints.set_zero(global_y_translation);
        // nullspace.basis.clear();      
        // nullspace.basis.push_back(global_rotation);
        // nullspace.basis.push_back(global_x_translation);
        // nullspace.basis.push_back(global_y_translation);
        // nullspace.orthogonalize();

        // // printf("Remove null space from solution: \n");
        // nullspace.remove_nullspace(solution);

        // constraints.distribute(solution);

        // solution.print(std::cout);
        print_mean_value();
    }

        // //Print small examples to illustrate.
        // printf("printing system matrix \n");
        // FullMatrix<double> tmp;
        // tmp.copy_from(system_matrix);
        // system_matrix.print(std::cout);
        // printf("Now print right hand side: \n");
        // system_rhs.print(std::cout);
        // printf("Global rotation \n");
        // global_rotation.print(std::cout);
        // printf("Global x \n");
        // global_x_translation.print(std::cout);
        // printf("Global y \n");
        // global_y_translation.print(std::cout);
        // printf("Checking if null space is orthogonal: \n");
        // printf("X and Y: %4f " , global_x_translation*global_y_translation);
        // printf("X and R: %4f " , global_x_translation*global_rotation);
        // printf("R and Y: %4f " , global_rotation*global_y_translation);    
        // constraints.condense(global_rotation);
        // constraints.condense(global_x_translation);
        // constraints.condense(global_y_translation);

  }


  template <int dim>
  void ElasticProblem<dim>::refine_grid()
  {
    Vector<double> cellwise_errors(triangulation.n_active_cells());
    QGauss<dim>    quadrature(fe.degree + 1);

    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      ExactSolution<dim>(),
                                      cellwise_errors,
                                      quadrature,
                                      VectorTools::L2_norm);

    Vector<double> exact(dof_handler.n_dofs());
    VectorTools::interpolate(dof_handler, 
                              ExactSolution<dim>(),
                              exact);

    // printf("Remove null from exact: \n");
    // nullspace.remove_nullspace(exact);
    // constraints.distribute(exact);
    // Vector<double> error_term = exact - solution;
    // nullspace.remove_nullspace(error_term);
    // constraints.distribute(error_term);

    error_u = // error_term.l2_norm();
      VectorTools::compute_global_error(triangulation,
                                        cellwise_errors,
                                        VectorTools::L2_norm);


    GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                    cellwise_errors,
                                                    0.3,
                                                    0.03);

    output_table.add_value("cells", triangulation.n_active_cells());
    output_table.add_value("error", error_u);
    output_table.add_value("MeanValue", mean_value);

    triangulation.execute_coarsening_and_refinement();
    // triangulation.refine_global();
  }

  template <int dim>
  void ElasticProblem<dim>::output_results(const unsigned int cycle) const
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
 
    std::vector<std::string> solution_names(dim, "displacement");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    interpretation(dim,
                   DataComponentInterpretation::component_is_part_of_vector);

    data_out.add_data_vector(dof_handler,
                              solution,
                              solution_names,
                              interpretation);


    Vector<double> exact(dof_handler.n_dofs());
    VectorTools::interpolate(dof_handler, 
                              ExactSolution<dim>(),
                              exact);

    // printf("Remove nullspace from exact. \n");
    // nullspace.remove_nullspace(exact);
    // constraints.distribute(exact);
    // exact.print(std::cout);

    data_out.add_data_vector(dof_handler,
                             exact,
                              "exact",
                              interpretation);

    Vector<double> error_term = exact - solution;
    
    // nullspace.remove_nullspace(error_term);
    // constraints.distribute(error_term);
    // printf("error after projection is %f \n", error_term.l2_norm());
    // error_term.print(std::cout);

    data_out.add_data_vector(dof_handler,
                              error_term, 
                              "Error",
                              interpretation);


    // data_out.add_data_vector(dof_handler,
    //                           global_rotation, 
    //                           "rot_int",
    //                           interpretation);


    // data_out.add_data_vector(dof_handler,
    //                           global_x_translation, 
    //                           "x_int",
    //                           interpretation);


    // data_out.add_data_vector(dof_handler,
    //                           global_y_translation, 
    //                           "y_int",
    //                           interpretation);


    // data_out.add_data_vector(dof_handler,
    //                           nullspace.basis[0], 
    //                           "rot_null",
    //                           interpretation);

    // data_out.add_data_vector(dof_handler,
    //                           nullspace.basis[1], 
    //                           "x_null",
    //                           interpretation);

    // data_out.add_data_vector(dof_handler,
    //                           nullspace.basis[2], 
    //                           "y_null",
    //                           interpretation);

    data_out.build_patches();

    std::ofstream output("solution-" + std::to_string(cycle) + ".vtk");
    data_out.write_vtk(output);
  }


  template <int dim>
  void ElasticProblem<dim>::run()
  {
    for (unsigned int cycle = 0; cycle < 8; ++cycle)
      {
        std::cout << "Cycle " << cycle << ':' << std::endl;

        if (cycle == 0)
          {
            GridGenerator::hyper_cube(triangulation, 0, 1);
            // Colorize will setup atuo.
            triangulation.refine_global(2);
            if (true)
            for (const auto &cell : triangulation.cell_iterators())
            {
              for (const auto &face : cell->face_iterators())
                {
                  // setup boundary id.
                  if (face-> at_boundary()){
                    const auto center = face->center();
                    if ((std::fabs(center(0) - (0.0)) < 1e-16) ){
                      // Left boundary.
                      face->set_boundary_id(1);
                    } else if ((std::fabs(center(0) - (1.0)) < 1e-16 )){
                      // Right boundary.
                      face->set_boundary_id(3); 
                    }else if ((std::fabs(center(1) - (1.0)) < 1e-16 )){
                      // top boundary.
                      face->set_boundary_id(2); 
                    }else if ((std::fabs(center(1) - (0.0)) < 1e-16 )){
                      // bottom boundary.
                      face->set_boundary_id(4); 
                    }
                    
                  }
                }

            }

          }
        else
          refine_grid();
          
        // std::cout << "   Number of active cells:       "
        //           << triangulation.n_active_cells() << std::endl;

        setup_system();

        // std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
        //           << std::endl;

        assemble_system();
        
        solve();
        output_results(cycle);




      }


    output_table.set_precision("error", 6);
    output_table.set_precision("MeanValue", 6);
    output_table.write_text(std::cout);
  }
}

int main()
{
  try
    {
      dealii::deallog.depth_console(99);
      Step8::ElasticProblem<2> elastic_problem_2d;
      elastic_problem_2d.run();
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
    }

  return 0;
}
