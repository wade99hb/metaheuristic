import numpy as np
import random
import math 
import matplotlib.pyplot as plt

def encode(x, len_bit):
    # 실수를 소수점 아래 4자리까지 고려하여 정수로 변환
    scaled_x = int(x * 10000)  # x를 10000배 하여 소수점 아래를 정수로 만듦

    # 정수를 이진수 문자열로 변환
    binary_str = bin(abs(scaled_x))[2:]  # 이진수로 변환하고, '0b' 접두사 제거

    # 음수일 경우를 위한 2의 보수 처리 (선택적)
    if x < 0:
        binary_str = bin((1 << len_bit) - int(binary_str, 2))[2:].zfill(len_bit) #zfill 메서드는 문자열이 특정 길이(여기선 len_bit)가 되도록 문자열 앞에 '0’을 채워 모든 비트가 사용되게함
        #2의 len_bit 승을 계산, binary_str이라는 2진수 문자열을 십진수로 해석하여 정수로 변환, 2의 len_bit 승에서 binary_str을 뺌. x의 2의 보수를 계산하는 과정
        
    # 필요한 경우 앞을 '0'으로 채워 비트 길이를 맞춤
    binary_str_padded = binary_str.zfill(len_bit)

    # 이진수 문자열을 numpy 배열로 변환
    binary_array = np.array(list(binary_str_padded), dtype=int) #np.array는 리스트나 튜플 등의 iterable 객체를 NumPy 배열로 변환
    #이진수 문자열을 리스트로 변환하고, 이를 np.array() 함수를 사용하여 numpy 배열로 변환합니다. 이 배열은 정수형(int) 데이터를 포함한다.
    return binary_array


def decode(binary_array):
    # numpy 배열을 문자열로 변환
    binary_str = ''.join(binary_array.astype(str))
    
    # 최상위 비트가 1이면 음수, 0이면 양수
    if binary_str[0] == '1': #음수인 경우
        # 2의 보수를 취하여 음수값 계산
        # 모든 비트를 반전시킨 후, 1을 더함
        inverse_binary_str = ''.join('1' if x == '0' else '0' for x in binary_str) #반전시키기
        decimal = -(int(inverse_binary_str, 2)+1)
    else: 
        # 양수일 경우 직접 10진수로 변환
        decimal = int(binary_str, 2) #예를 들어, int('1010', 2)는 이진수 문자열 '1010’을 십진수 정수 10으로 변환
    
    decimal = round(decimal*(10**-4),4) #0.0001, 십진수를 10000으로 나누어 소수점 아래 4자리까지 고려한 실수로 반올림 변환한다
    return decimal

# 초기해 생성
def initial(bit1, bit2): 
    x1=np.random.uniform(-3, 12.1)
    encoded1=encode(x1, bit1)
    
    x2=np.random.uniform(4.1, 5.8)
    encoded2=encode(x2, bit2)
    
    return encoded1, encoded2

# 이웃해 생성
def neighborhood(encoded1, encoded2, bit1, bit2):
    swap1, swap2 = np.random.choice(range(bit1), 2, replace=False)
    swap3, swap4 = np.random.choice(range(bit2), 2, replace=False)
    
    neighborhood1 = encoded1.copy()
    neighborhood2 = encoded2.copy()
    
    neighborhood1[swap1], neighborhood1[swap2] = neighborhood1[swap2], neighborhood1[swap1]
    neighborhood2[swap3], neighborhood2[swap4] = neighborhood2[swap4], neighborhood2[swap3]
    
    return neighborhood1, neighborhood2

# 목적함수
def E(x1, x2):  
    return 21.5 + x1 * np.sin(4 * np.pi * x1) + x2 * np.sin(20 * np.pi * x2)

# 해를 수락할지 결정하는 함수
def yes_or_no(p):  
    u = np.random.random()
    if p > u:
        return True
    else:
        return False
    
    
# 알고리즘의 메인 루프
def algorithm():
    N=10000
    T=10
    bit1, bit2= 18, 17 
    best_solutions = [] #찾은 최적해를 모은 리스트
    best_solution = 0  #현재까지 찾은 최적해이고 0부터 초기화. 천번동안 큰 값이 나올때마다 업데이트
    
    for i in range(N):
        initial_sol1, initial_sol2= initial(bit1, bit2) #초기해생성
        neighbor_sol1, neighbor_sol2 = neighborhood(initial_sol1, initial_sol2, bit1, bit2) #이웃해 생성
        
        decode_in1, decode_in2 = decode(initial_sol1), decode(initial_sol2) #초기해 디코딩
        decoded_nei1, decoded_nei2 = decode(neighbor_sol1), decode(neighbor_sol2) #이웃해 디코딩
        
        delta = E(decode_in1, decode_in2) - E(decoded_nei1, decoded_nei2) #델타=초기해-이웃해
        
        if delta < 0: 
            initial_sol1, initial_sol2 = neighbor_sol1, neighbor_sol2 #초기해=이웃해로 업데이트
            best_solution = max(best_solution, E(decode_in1, decode_in2))  #찾아놓은 최적해, 목적함수에 새로운 해 넣은 값 중 큰값으로 최적해 업데이트
            
        else: #delta >= 0 
            p = np.exp(-delta / T) #수락확률
            if yes_or_no(p): #true나 false 반환, if yes_or_no(p):는 if yes_or_no(p) == True:와 동일한 의미
                initial_sol1, initial_sol2 = neighbor_sol1, neighbor_sol2
                
        T = 0.95 * T #쿨링스케줄, a=0.95
        
        best_solutions.append(best_solution)
    return best_solutions

# 시각화
best_solutions = algorithm()
plt.plot(best_solutions)
plt.title('Finding Global Maximum')
plt.xlabel('Iteration')
plt.ylabel('Best Solution')
plt.show()
print('최댓값:', max(best_solutions))
