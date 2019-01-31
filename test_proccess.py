import glob
import func
import os.path as op

files = []
dog_folder_name = "いぬ"
files += glob.glob("data/いぬ/*.jpg")
files += glob.glob("data/ねこ/*.jpg")

# dog_folder_name  = "dog"
# files += glob.glob("data/dog/train/*.jpg")
# files += glob.glob("data/cat/train/*.jpg")
result = func.test_process()

# total = 0
# collect = 0
# for imgname in files :
#     print(imgname)
#     lavel = "dog" if dog_folder_name in imgname else "cat"
#     result = func.test_process(imgname)
#     if result == lavel:
#         collect += 1
#     total += 1


# print(f"試験数: {total}")
# print(f"正解数: {collect}")
# print(f"正解率: {collect/total}")
