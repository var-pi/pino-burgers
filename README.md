This repo provides an example implementation of a FNO/PINO for a burgers equation. Adding initial condition loss reduced the MSE by a factor of 5. The PDE loss calculated with finite differences on the other side considerably hinders the training probably due to coarse resolution of the grid (which coases large discretiyaion errors). A solution would be to upsample the inputs for PDE loss. This however wouold requre a custom trainer.

![Figure 1](figures/Figure_1.png)
![Figure 2](figures/Figure_2.png)
![Figure 3](figures/Figure_3.png)
![Figure 4](figures/Figure_4.png)
