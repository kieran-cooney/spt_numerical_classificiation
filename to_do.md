* Start calculations

* How to perform calculations?
    * If using iDMRG, may need to add two sites each time, as the parity of the number of sites is apparently important.

* Which perturbation to apply to model?
    * Crucial that the Z2 x Z2 SPT phase is kept intact.
        * So the perturbation must preserve the symmetry...?
        * How to vary to get different projective representations, i.e. cohomology classes?
    * Boundary conditions, periodic or open ended? Does it matter?
    * Even or odd number of sites?
    * Transverse field (one body), see "QI meets QM" book.
        * Can calculate topological entanglement entropies according to same book (chapter 5).
    * Can have seperate parameters for odd and even sites. See Kemp, Yao and Laumann paper for a very general Hamiltonian.
    * "Cluster-Ising model (CIM)", a 3-body XZX cluster term and a two body Ising perturbtation in the y direction.
        * Leads to competing "cluster" and "Ising" phases.
        * Many nearly analytic expressions obtainable, as there is a clean expression for the partition function.
    * "Cluster XY model", seems to be a generalization of the cluster Ising model. Adds a one body z term and two body x and y terms.

* Verify calculations, what to check from model?
    * For cluster-ising model, many properties
    * String order parameter
    * Entanglement entropy spectrum?
    * Can't detect degeneracy...
    * Correlation length

* Theory seems to suggest that the unitary will only ever need to be defined on the one site at the boundary. Investigate and justify, both theoretically and numerically.

* What about applying a non-uniform transverse magnetic field?

# Gatchas
* The "local" unitaries applied to undo the finite symmetry action could diverge in area as the critical point is approached. This is annoying, as the critical point is precisely the area we would like to investigate...!

# Other models
* Haldane phase (?), AKLT
* Kitaev chain (fermionic)

# Questions
* Will the phase calculation become more difficult for parameters close to a phase transition?

# To read
* DMRG and iDMRG algorithms
