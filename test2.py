import numpy as np
import matplotlib.pyplot as plt

def encode(x, len_bit):
    scaled_x = int(x * 10000)
    binary_str = bin(abs(scaled_x))[2:]

    if x < 0:
        binary_str = bin((1 << len_bit) - int(binary_str, 2))[2:].zfill(len_bit)

    binary_str_padded = binary_str.zfill(len_bit)
    binary_array = np.array(list(binary_str_padded), dtype=int)
    return binary_array

def decode(binary_array):
    binary_str = ''.join(binary_array.astype(str))
    if binary_str[0] == '1':
        inverse_binary_str = ''.join('1' if x == '0' else '0' for x in binary_str)
        decimal = -(int(inverse_binary_str, 2) + 1)
    else:
        decimal = int(binary_str, 2)

    decimal = round(decimal * (10**-4), 4)
    return decimal

def firstsun(len_bit1, len_bit2):
    x1 = np.random.uniform(-3.0, 12.1)
    x2 = np.random.uniform(4.1, 5.8)
    encoded_value1 = encode(x1, len_bit1)
    encoded_value2 = encode(x2, len_bit2)
    return encoded_value1, encoded_value2

def neighborhood(encoded_value1, encoded_value2, len_bit1, len_bit2):
    randomnum1 = np.random.randint(0, len_bit1)
    randomnum2 = np.random.randint(0, len_bit2)
    neighborhood_value1 = encoded_value1.copy()
    neighborhood_value2 = encoded_value2.copy()
    neighborhood_value1[randomnum1] = 1 - neighborhood_value1[randomnum1]
    neighborhood_value2[randomnum2] = 1 - neighborhood_value2[randomnum2]
    return neighborhood_value1, neighborhood_value2

def maximize(decoded_value1, decoded_value2):
    return 21.5 + decoded_value1 * np.sin(4 * np.pi * decoded_value1) + decoded_value2 * np.sin(20 * np.pi * decoded_value2)

def yes_or_no(p): #수락 결정 함수
    return np.random.rand() < p

def master(N, Tk, cooling_rate, len_bit1, len_bit2):
    best_solutions = []
    best_energies = []
    T = Tk

    encoded_value1, encoded_value2 = firstsun(len_bit1, len_bit2) # 초기해 엔코더 
    decoded_value1, decoded_value2 = decode(encoded_value1), decode(encoded_value2)#초기해 디코더
    best_energy = maximize(decoded_value1, decoded_value2) # 초기해 목적함수 적용
    current_energy = best_energy

    for i in range(N):
        neighborhood_value1, neighborhood_value2 = neighborhood(encoded_value1, encoded_value2, len_bit1, len_bit2) #이웃해 생성
        decoded_n1, decoded_n2 = decode(neighborhood_value1), decode(neighborhood_value2) #이웃해 디코더
        neighbor_energy = maximize(decoded_n1, decoded_n2) # 이웃해 목적함수 적용

        delta = current_energy - neighbor_energy 

        if delta < 0 or yes_or_no(np.exp(-delta / T)):
            decoded_value1, decoded_value2 = decoded_n1, decoded_n2
            current_energy = neighbor_energy

            if current_energy >= best_energy: #delta >= 0
                best_energy = current_energy

        best_solutions.append((decoded_value1, decoded_value2))
        best_energies.append(best_energy)

        T *= cooling_rate

    return best_solutions, best_energies


N = 1000
Tk = 10
cooling_rate = 0.95
len_bit1, len_bit2 = 18, 17

best_solutions, best_energies = master(N, Tk, cooling_rate, len_bit1, len_bit2)

print("최적 해:", best_solutions[-1])
print("최적 에너지:", best_energies[-1])

plt.plot(best_energies)
plt.title('Simulated')
plt.xlabel('Iteration')
plt.ylabel('Best_Energy')
plt.show()
