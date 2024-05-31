* Stick with simple gradient descent, but update learning rate intelligently.
    * If loss not changing appreciably, lower learning rate and unitarize.
* Can have matrix elements initialize to small value initially with relative ease. Should re-initialize in this case.
* There are many redundancies we are making assumtpions for which should be checked:
    * How many sites should the segment have? If the segment has too many sites, does it matter which sub-segment we operate on to compute expectations?

* I want to look up Riemannian optimization for gradient descent on manifolds.
    * https://pymanopt.org/docs/stable/manifolds.html#pymanopt.manifolds.group.UnitaryGroup

* Infinite/finite? Proving to be a major headache for me.
    * Let's do a proof of concept in a small finite system first, just to see if it's working.
    * But ideally we should use infinite MPS. I'm using tenpy to do this currently, and it can be a bit of a pain. Not sure how much support I'm going to get either.
    * Moving to Julia fulltime might be a better choice. But ideally language agnostic would be best.
    * The problem with iDMRG is extracting the environments. How to do properly? MPSKit and ITensor offer some choices, but not super well supported. Seem to be my best options. So Julia is in fact inevitable in some sense.


* Implement manifold optimization. Another point for Julia.

* The gradient has a problem in that it may vanish if the matrix element itself is small or zero. In fact, the set of 0's probably (?) divides the space in two, so for half of the initial conditions it will have to pass through this space, and so never get there! (Think of the similar problem on the circle.) How to address? Need a different cost function? Or use a "swarm" approach...

* The operator norm of the finite segment symmetry operator (when viewed as a transfer matrix) seems to be very close to 2. Why?