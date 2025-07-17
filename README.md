# PPP project course - MPI parallel heat solver
- Author: Ondřej Vlček

![Heat simulation.](images/heat.jpg)

## File structure
    3rdparty           - Third party libraries
    doc                - Source files for student assignment docummentation
    eval_scripts       - Evaluation scripts 
    scripts            - Test scripts for students (*.sh, *.py, *.flt)
    sources            - Project source files

## Assignment package preparation
Following command will generate package that can be published as assignment for students. The `assignment.zip` file will
be placed in `build` directory.

    $ cd sources_solution
    $ mkdir build && cmake ..
    $ make assignment

## Results

### Measured scaling
| 1D | 2D |
| :-: | :-: |
| ![1D scaling](images/ppp_scaling_hybrid_1D_p2p.svg) | ![2D scaling](images/ppp_scaling_hybrid_p2p.svg) |

### Measured efficiency
| 1D | 2D |
| :-: | :-: |
| ![1D efficiency](images/ppp_efficiency_hybrid_1D_p2p.svg) | ![2D efficiency](images/ppp_efficiency_hybrid_p2p.svg) |

### Computational load

| 2D processes | 2D processes + threads |
| :-: | :-: |
| ![2D processes only](images/2DhaloComp.png) | ![2D processes and threads](images/2DhaloThreads.png)  |