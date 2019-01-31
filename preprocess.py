import func
import os.path as op



i = 0
for filename in func.train_file_names :
    # ディレクトリ名入力
    while True :
        dirname = input(">>「" + func.CLASSES[i] + "」の画像のあるディレクトリ ： ")
        if op.isdir(dirname) :
            break
        print(">> そのディレクトリは存在しません！")

    # 関数実行
    func.pre_process(dirname, filename, var_amount=3)
    i += 1

i = 0

for filename in func.valid_file_names :
    # ディレクトリ名入力
    while True :
        dirname = input(">>「" + func.CLASSES[i] + "」の画像のあるディレクトリ ： ")
        if op.isdir(dirname) :
            break
        print(">> そのディレクトリは存在しません！")

    # 関数実行
    func.pre_process(dirname, filename, var_amount=3)
    i += 1

i = 0

for filename in func.test_file_names :
    # ディレクトリ名入力
    while True :
        dirname = input(">>「" + func.CLASSES[i] + "」の画像のあるディレクトリ ： ")
        if op.isdir(dirname) :
            break
        print(">> そのディレクトリは存在しません！")

    # 関数実行
    func.pre_process(dirname, filename, var_amount=3)
    i += 1