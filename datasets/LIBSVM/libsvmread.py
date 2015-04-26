# -*- coding: utf-8 -*-
import numpy as np

if __name__ == '__main__':
    print 'hello'
    dim = 13 + 1 # 次元: ラベルの分だけ+1する
    num = 270 # データ数
    # 読み込むファイル名を指定
    f = open('heart_scale')
    data1 = f.read()  # ファイル終端まで全て読んだデータを返す
    f.close()
    print type(data1) # 文字列データ
 
    lines1 = data1.split('\n')
    print type(lines1)
    mat = [lines1[i].rstrip().split() for i in range(len(lines1))]

    dataset = np.zeros([num, dim])

    for row in range(num):
        for j in range(len(mat[row])):
            if j == 0: dataset[row, j] = int(mat[row][j])
            else:
                col, val = mat[row][j].split(':')
                col = int(col)
                val = float(val)
                dataset[row, col] = val

    # 出力するファイル名や形式を指定
    np.savetxt('heart_scale.csv', dataset, delimiter=',', fmt='%s')
