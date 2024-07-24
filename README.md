#Metaheuristic Project
import numpy as np
import matplotlib.pyplot as plt

def encode(x, len_bit):
    # 실수를 소수점 아래 4자리까지 고려하여 정수로 변환
    scaled_x = int(x * 10000)  # x를 10000배 하여 소수점 아래를 정수로 만듦

    # 정수를 이진수 문자열로 변환
    binary_str = bin(abs(scaled_x))[2:]  # 이진수로 변환하고, '0b' 접두사 제거

    # 음수일 경우를 위한 2의 보수 처리 (선택적)
    if x < 0:
        binary_str = bin((1 << len_bit) - int(binary_str, 2))[2:].zfill(len_bit) #zfill 메서드는 문자열이 특정 길이(여기선 len_bit)가 되도록 문자열 앞에 '0’을 채워 모든 비트가 사용되게함
        #2의 len_bit 승을 계산, binary_str이라는 2진수 문자열을 십진수로 해석하여 정수로 변환, 2의 len_bit 승에서 binary_str을 뺀 후 1을 뺌??????????????. x의 2의 보수를 계산하는 과정
        
    # 필요한 경우 앞을 '0'으로 채워 비트 길이를 맞춤
    binary_str_padded = binary_str.zfill(len_bit)

    # 이진수 문자열을 numpy 배열로 변환
    binary_array = np.array(list(binary_str_padded), dtype=int) #np.array는 리스트나 튜플 등의 iterable 객체를 NumPy 배열로 변환
    #이진수 문자열을 리스트로 변환하고, 이를 np.array() 함수를 사용하여 numpy 배열로 변환합니다. 이 배열은 정수형(int) 데이터를 포함합니다.
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

def firstsun (len_bit1, len_bit2) : #초기해
    x1 = np.random.uniform(-3.0,12.1) #-3.0 ~ 12.1
    x2 = np.random.uniform(4.1,5.8) #4.1~5.8
    encoded_value1 = encode(x1, len_bit1) #인코딩
    encoded_value2 = encode(x2, len_bit2) #인코딩
    return encoded_value1, encoded_value2

def neighborhood(encoded_value1, encoded_value2, len_bit1, len_bit2): # 이웃해
    
    randomnum1 = np.random.randint(0, len_bit1)  # x1 비트수 내에서 랜덤 위치 추출
    randomnum2 = np.random.randint(0, len_bit2)  # x2 비트수 내에서 랜덤 위치 추출
    
    neighborhood_value1 = encoded_value1.copy()  # 전체 배열을 복사
    neighborhood_value2 = encoded_value2.copy()  # 전체 배열을 복사
    
    neighborhood_value1[randomnum1] = 1 - neighborhood_value1[randomnum1]  # 1의 보수 변환
    neighborhood_value2[randomnum2] = 1 - neighborhood_value2[randomnum2]  # 1의 보수 변환
    
    return neighborhood_value1, neighborhood_value2

def decodefirstsun(encoded_value1,encoded_value2) : #디코딩 초기해
    decoded_f1 = decode(encoded_value1) #디코딩
    decoded_f2 = decode(encoded_value2) 
    return decoded_f1, decoded_f2

def decodeneighbor(neighborhood_value1, neighborhood_value2) : #디코딩 이웃해
    decoded_value1 = decode(neighborhood_value1) #디코딩
    decoded_value2 = decode(neighborhood_value2) #디코딩
    return decoded_value1 , decoded_value2

def maximize(decoded_value1, decoded_value2): #최대화 (목적함수)
    return 21.5 + decoded_value1*np.sin(4*np.pi*decoded_value1) + decoded_value2*np.sin(20*np.pi*decoded_value2) #최대화
    

def argorism(firsttemp, coolingtemp, deathtemp):
    encoded_value1, encoded_value2 = firstsun(18, 17)
    decoded_f1, decoded_f2 = decodefirstsun(encoded_value1, encoded_value2)
    current_solution = (decoded_f1, decoded_f2)
    current_energy = maximize(*current_solution) #튜플활용 최대화
    best_solution = current_solution
    best_energy = current_energy


    best_solution = current_solution
    best_energy = current_energy

    temp = firsttemp
    qksqhr = 1000
    neighbor_list = []
    best_list = []
    energy_list = []
    # 반복 횟수 계산
   
    for i in range(qksqhr):
        if temp <= deathtemp:
            break

        neighborhood_value1, neighborhood_value2 = neighborhood(encoded_value1, encoded_value2, 18, 17)
        decoded_n1, decoded_n2 = decodeneighbor(neighborhood_value1, neighborhood_value2)
        neighbor_energy = maximize(decoded_n1, decoded_n2)

        delta = neighbor_energy - current_energy

        if delta > 0 or np.exp(delta / temp) > np.random.rand(): #수락확률
            current_solution = (decoded_n1, decoded_n2)
            current_energy = neighbor_energy
            encoded_value1, encoded_value2 = neighborhood_value1, neighborhood_value2

        if current_energy > best_energy: # Y > X
            good_solution = current_solution
            best_energy = current_energy
            
            neighbor_list.append(neighbor_energy)
            best_list.append(good_solution)
            energy_list.append(current_energy)

        temp *= coolingtemp

    return good_solution, best_energy, neighbor_list, best_list,energy_list

good_solution, best_energy, neighbor_list, best_list, energy_list  = argorism(1000,0.9,1)
print(good_solution, best_energy, neighbor_list, best_list, energy_list)
plt.plot(energy_list)
plt.xlabel('frequently')
plt.ylabel('energy')
plt.title('argorism')
plt.show()
