# Numerical Experiments for Robust Support Vector Machines

## Required Libraries
* CPLEX
    - Used to solve subproblems.
* Numpy
* Matplotlib
    - Used to draw figures.
* scikit-learn
    - Used to compute kernel matrix.
* Pandas
  - Used for data handling

## Run script

```sh
python expt_mnist.py
```

### Memo
* カーネル行列をそのままメモリに入れるとかなり辛い.
  - #sample = 10000 くらいでメモリの限界.
  - 計算では目的関数の 2 次の項をセットする部分がボトルネック.
  - Ramp Loss SVM だと #sample = 9000 で約 10 min, #sample = 8000 で 5 min.