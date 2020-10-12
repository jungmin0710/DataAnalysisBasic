##numpy basic

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns 
import statsmodels as sm

my_arr = np.arange(1000000)
my_list = list(range(1000000))

# 실행 속도차이 비교(쥬피터에서 사용할 것) => 넘파이 사용하는 것이 더 빠름
%time for _ in range(10): my_arr2 = my_arr *2
%time for _ in range(10): my_list2 = [x * 2 for x in my_list]

#dtype 찍어보기
arr1 = np.array([1,2,3], dtype = np.float64)
arr2 = np.array([1,2,3], dtype = np.int32)

arr1.dtype
arr2.dtype


#array배열의 산술연산

arr = np.array([[1,2,3],[4,5,6]])
arr

arr * arr
arr - arr
1/arr
arr ** 0.5

#array배열의 색인과 자르기(파이썬 기본리스트와 동일)
arr = np.arange(10)
arr

arr[5]
arr[5:8]
arr[5:8] = 12
arr
arr_slice = arr[5:8]
arr_slice
arr_slice[1] = 12345
arr
arr_slice[:]=64
arr
arr2d = np.array([[1,2,3],[4,5,6],[7,8,9]])
arr2d[2]
arr2d[0][2]
arr2d[0,2]

#boolean Indexing
names = np.array(['Bob', 'Joe', 'Will', 'Bob','Will','Joe','Joe'])
data = np.random.randn(7, 4)
names
data
names == 'Bob'
data[names == 'Bob']
