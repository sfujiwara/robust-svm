# Numerical Experiments for Robust Support Vector Machines

## Required Libraries

* CPLEX
* Numpy
* scikit-learn
* Pandas

## Set Up for Red Hat Linux

基本的に管理者権限を必要とせず, 環境を汚さないような手順を記載します.

### Required Packages for Python

`requirements.txt` に必要なパッケージを記述したので,  pip (Python のパッケージ管理ツール) を使って以下のようにまとめてインストールすることができます.

```sh
pip install -r requirements.txt --user
```

pip が無い場合や, NumPy や SciPy のインストールが依存関係を解決できずに失敗する場合は連絡をお願いします.
学内に Python を使用している人がいる場合は問題なく通ると思いますが, そうでない場合は管理者権限が無いとかなり面倒なのでやり方を考えます.

### CPLEX

CPLEX の Python API を呼び出せるように, CPLEX がインストールされたディレクトリに置かれている `setup.py` を実行します.

```sh
python setup.py install --user
```

Ubuntu の場合は `/opt/ibm/ILOG/CPLEX_Studio126/cplex/python/x86-64_linux/setup.py` にありましたが, Red Hat の場合や CPLEX のバージョン次第では多少違いがあるかもしれません.


## Run script

MNIST データセット用いた実験用スクリプトを以下のように実行します.
MNIST データセットのダウンロードも自動で行われます.

```sh
python expt_mnist.py
```

## Output Files

* 最後までスクリプトが走ると `results/mnist` 以下に実験結果の csv ファイルが出力されます.
* 実験の途中経過の log を `logs/expt_mnist.log` に出力しています
* エラーが発生した場合 `logs/mnist_stderror.txt` にエラーメッセージが出力されます
  - こちらはきちんとした log ではなく, 標準出力されるエラーメッセージをそのまま流しているだけです
