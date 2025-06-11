## Mixed_Curvature_VRPs
This is the code repo of the implementation for the ICML 2025 paper: 

*A Mixed-Curvature based Pre-training Paradigm for Multi-Task Vehicle Routing Solver*
![VD (1)](https://github.com/user-attachments/assets/b42447c2-b738-4aef-97b4-5e011f909f48)

It is a neural solver established on the geometric curvature spaces where hidden features are processed through the non-Euclidean spaces, encouraging the model to capture the un
derlying geometric properties of each instance. We modify multi-task solvers like POMO-MTL and MVMoE(-L) into this geometric learning settings. We provide their implementations in files MTLModel_Mixed.py, MOEModel_Mixed.py and MOEModel_Light_Mixed.py, respectively.

## How to Run the Codes

    Default Settings: --problem_size=50 --pomo_size=50 --gpu_id=0

1. To run POMO-MTL-Mixed
   
       python train.py --problem=Train_ALL --model_type=MTL_Mixed
   
3. To run MVMoE-Mixed
   
       python train.py --problem=Train_ALL --model_type=MOE_Mixed
   
5. To run MVMoE-Light-Mixed
   
       python train.py --problem=Train_ALL --model_type=MOE_LIGHT_Mixed

## Dependency
Python >= 3.9

Pytorch >= 2.0.0

Geoopt >= 0.4.0

CUDA >= 11.8

## Acknowledgments
We want to express our sincere thanks to the following works:

[POMO: Policy Optimization with Multiple Optima for Reinforcement Learning](https://github.com/yd-kwon/POMO)

[MVMoE: Multi-Task Vehicle Routing Solver with Mixture-of-Experts](https://github.com/RoyalSkye/Routing-MVMoE)

[RouteFinder: Towards Foundation Models for Vehicle Routing Problems](https://github.com/ai4co/routefinder)

[RL4CO: an Extensive Reinforcement Learning for Combinatorial Optimization Benchmark](https://github.com/ai4co/rl4co)

[Geoopt: Riemannian Optimization in PyTorch](https://github.com/geoopt/geoopt)

[Mixed-Curvature Transformers for Graph Representation Learning](https://openreview.net/forum?id=DFnk58DwTE) 

[Community detection on networks with ricci flow](https://github.com/saibalmars/GraphRicciCurvature)

## Citations

  <div style="position: relative">
    <button onclick="navigator.clipboard.writeText(document.getElementById('bibtex-cite').innerText)" style="position: absolute; top: 4px; right: 4px;"></button>
    <pre id="bibtex-cite"><code>
@inproceedings{liu2025mixed,
  title     = {A Mixed-Curvature based Pre-training Paradigm for Multi-Task Vehicle Routing Solver},
  author    = {Suyu Liu and Zhiguang Cao and Shanshan Feng and Yew-Soon Ong},
  booktitle = {International Conference on Machine Learning},
  year      = {2025}
}
    </code></pre>
  </div>


