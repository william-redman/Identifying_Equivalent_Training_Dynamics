# Identifying_Equivalent_Training_Dynamics
This repository hosts the code for the ["Identifying Equivalent Training Dynamics"](https://openreview.net/forum?id=bOYVESX7PK&noteId=9e5HVNLBta) Redman et al. NeurIPS (2024) Spotlight paper. Code for two of the experiments used in the paper (1. training dynamics of online mirror, online gradient, and bisection method; 2. training dynamics of fully connected neural networks) are provided in their respective folders. More code will uploaded, so check back regularly! Also feel free to reach out to wredman4@gmail.com for any questions on how to use the Kooopman operator theoretic framework for studying training dyanmics. 

## Online mirror online gradient descent
This folder provides code for studying the training dynamics of online mirror descent and online gradient descent, and compares them to the dynamics of the bisection method. Run ```online_mirror_online_gradient_descent_main.py``` to replicate the results presented in Fig. 2 of the paper. Run ```online_mirror_online_gradient_descent_computing_significance.py``` to perform the randomized shuffle control to identify signifcance. Results are saved into the ***Results** folder and figures saved into the **Figures** folder. For completeness, we have included the saved Koopman eigenvalues and figures from our own analysis. 

### Fully connected neural networks
This folder provides code for studying the training dynamics of FCNs, trained on MNIST, for varying widths. Run ```FCN_MNIST_main.py``` to generate your own training trajectories. Vary the parameter *h* to manipulate the width of the FCN. The weight trajectories will get saved into the **Results** folder. Run ```FCN_plotting_results.py``` to generate plots analogous to Fig. 3 of the paper. For completeness, we have also included the Koopman eigenvalues from our own analysis. Running ```FCN_plotting_wasserstein_distance.py``` will replicate the Fig. 3D-F. See the **Figures** folder for our saved figures.


