## Overview

This repository contains the implementation of the methods described in:

**Generalized TV--$\ell_p$ Structured Priors for Bayesian $T_1$ Mapping**\
*Disi Lin; Martin Berggren; Tommy Löfstedt*

This paper proposes an extended family of structured spatial priors that incorporates the total variation (TV) function with $\ell_p$ norms. 
The prior is proven to be proper and incorporated into a Bayesian regression framework to enable uncertainty quantification in $T_1$ mapping, with posterior inference performed using the No-U-Turn Sampler (NUTS).



------------------------------------------------------------------------

## Dependencies

pymc 5.11.0\
pytensor 2.20.0\
numpyro 0.15.0\
arviz 0.18.0\
matplotlib 3.8.4

------------------------------------------------------------------------

## Usage

### Run different methods to get results

``` bash
python mx_xxxx.py 
```

### Plot results

``` bash
python fig_pdfs.py
```

------------------------------------------------------------------------

## Contact
- disi.lin@umu.se; disilinumu@gmail.com
