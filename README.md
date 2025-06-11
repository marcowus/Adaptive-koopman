This is the code repository for the paper titled "**Adaptive Koopman Architectures for Control of Complex Nonlinear Systems**". 

The original paper can be found at https://journals.sagepub.com/eprint/4CDXT9UJZYPBE6MJMIQ6/full. 

An archive copy of the paper can be found at https://arxiv.org/abs/2405.09101.

If you use the code in the academic context, please cite:
- Singh R, Sah CK, Keshavan J. Adaptive Koopman embedding for robust control of nonlinear dynamical systems. The International Journal of Robotics Research. 2025;0(0). doi:10.1177/02783649251341907

This study presents an adaptive Koopman algorithm capable of responding to the changes in system dynamics online. The proposed framework initially employs an autoencoder-based neural network which utilizes input-output information from the nominal system to learn the corresponding Koopman embedding offline. Subsequently, we augment this nominal Koopman architecture with a feed-forward neural network that learns to modify the nominal dynamics in response to any deviation between the predicted and observed lifted states, leading to improved generalization and robustness to a wide range of uncertainties and disturbances as compared to contemporary methods.The proposed adaptive Koopman architecture is integrated within a Model Predictive Control (MPC) framework to enable optimal control of the underlying nonlinear system in the presence of uncertainties along with state and input constraints. 

![adaptation_block](https://github.com/Rajpal9/Adaptive-koopman/assets/90927685/9bcaec27-a618-40e6-bb59-6771908c5ec1)


  
