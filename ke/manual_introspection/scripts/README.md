#### Contains remaining ToDos for this file.

So far a selection of comparisons were conducted:

CIFAR10:
- [x] Comparisons between two models (first model to other)
    - [x] At different layers 
    - [x] At different depths 
    - [ ] With different loss dis - weights
    - [x] At multiple layers
    - [ ] With different loss metrics 
      - [x] ExpVar
      - [ ] L2Corr
      - [ ] LinCKA
    - [ ] Other Architectures ? 
        - [ ] ResNet34
        - [ ] ResNet101
- [ ] Comparisons between ensembles of models (first model to 5 others)
- [ ] Some table with numeric results?

CIFAR100:
- [ ] Check which trainings are lying around
  - [ ] (Basically all are ResNet18 if I am not mistaken)


- [ ] Visualization of similarity approximation only.

### More examples of Cohens Kappa to relative ensemble performance?
- [ ] Measure the average cohens kappa between models -- 
  - Similarly to a Heatmap -> Visualizes the ensemble similarity?
- [ ] Or maybe a table that shows the progression of ensemble performance for growing ensemble numbers
    - Do not only plot accuracy but also:
      - average new model accuracy
      - absolute ensemble performance
      - relative ensemble performance 
      - Cohens Kappa
      - Jensen Shannon Divergence
  
