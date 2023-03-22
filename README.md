# CompareLaplaceConstraint
- all_one : Condensing constraint of all ones in laplace matrix.
- fixing_points : fixing one random point and pass it into laplace matrix as constraint.
- int_u : interpolate \int_\Omega u = 0, store it as a vector, pass it into laplace matrix as constraint. (takes forever, need to be optimized)
- int_partial_u : same as previous one but only on boundary.

# Elastic Problem:

- ElasticityWithCurl : Interpolate translational null space as Laplace problem and interpolate curl. (Highly likely something wrong here, but I can't figure out a better way :(, takes forever to compute. )
- ElasticityBuiltIn: In 2d, fixing one point on both x and y direction, then pick another point and fix one direction. (The picture I shared with you come from this one)
- ElasticityRM: Remove all three constraint(haven't started yet).
