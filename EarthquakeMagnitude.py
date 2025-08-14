import os
import math
import numpy


def load_data(input_path, start_num, end_num):
    m0 = []
    for i in range(start_num, end_num):
        input_file_path = os.path.join(input_path, f'MR{i}.txt')
        if not os.path.exists(input_file_path):
            continue
        with open(input_file_path, 'r') as file:
            data = file.readline().strip().split()
            data = [float(x) for x in data]
            C = 0
            for j in range(0, len(data)):
                C = C + data[j] * 0.20006667
            m0.append(C)

    return m0


Input_path = r'/Nas/Liqy/EarthquakeSeismicNet/data_only_90/MR/'
Start_num = 1
End_num = 25526
Mw = []
M0 = load_data(Input_path, Start_num, End_num)

for i in range(0, len(M0)):
    mw = 2 * (math.log(M0[i], 10) - 9.1) / 3
    Mw.append(mw)

Mw_max = max(Mw)
Mw_min = min(Mw)
Mw_mean = numpy.mean(Mw)
Mw1 = numpy.float_(Mw)
m = numpy.sum(Mw1 > Mw_mean)
n = numpy.sum(Mw1 < Mw_mean)
print("Mw最大值的位置：", Mw.index(max(Mw)))
# Displaying the array
file = open("sample0_12000.txt", "w+")

# Saving the array in a text file
content1 = str(Mw)
content2 = str(Mw_max)
content3 = str(Mw_min)
content4 = str(Mw_mean)
file.write(content1)
file.write(content2)
file.write(content3)
file.write(content4)
file.close()

# Displaying the contents of the text file
file = open("sample0_12000.txt", "r")

content = file.read()

# print("Array contents in sample0_12000.txt: ", content)
file.close()

print("震级最大值：", Mw_max)
print("震级最小值：", Mw_min)
print("震级均值：", Mw_mean)
print("震级大于均值的个数：", m)
print("震级小于均值的个数：", n)
