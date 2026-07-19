import pathlib
import random

# modes available : normal, trap, noisy, all_random
MODE = "all_random"
NOISE = 1.00

def get_random_sequence(nb_bits):
    sequence = ""
    for i in range(nb_bits):
        sequence += str(random.randint(0, 1))
    return sequence

def get_symmetrical_sequence(nb_bits):
    half_size = nb_bits // 2
    half = ""
    for j in range(half_size):
        half += str(random.randint(0, 1))
    sequence = half
    if nb_bits % 2 == 1:
        sequence += str(random.randint(0, 1))
    reversed_half = half[::-1]
    sequence += reversed_half
    rotation = random.randint(0, nb_bits - 1)
    sequence = sequence[rotation:] + sequence[:rotation]
    return sequence

def get_trap_sequence(nb_bits, symmetrical_sequence):
    sequence = symmetrical_sequence
    i = random.randint(0, nb_bits - 1)
    while sequence[i] == sequence[(i + 1) % nb_bits]:
        i = random.randint(0, nb_bits - 1)
    sequence = list(sequence)
    t = sequence[i]
    sequence[i] = sequence[(i + 1) % nb_bits]
    sequence[(i + 1) % nb_bits] = t
    sequence = "".join(sequence)
    return sequence

def add_noise(sequence, noise):
    mask = [True] * len(sequence)
    for i in range(noise):
        j = random.randint(0, len(sequence) - 1)
        while mask[j] == False:
            j = random.randint(0, len(sequence) - 1)
        mask[j] = False
    sequence = list(sequence)
    for i in range(len(sequence)):
        if mask[i] == False:
            sequence[i] = str(random.randint(0, 1))
    sequence = "".join(sequence)
    return sequence

def is_symmetrical(sequence):
    half_size = len(sequence) // 2
    for i in range(len(sequence)):
        k = i
        l = i
        if len(sequence) % 2 == 0:
            k = i
            l = (i + 1) % len(sequence)
        else:
            k = i - 1
            l = (i + 1) % len(sequence)
        for j in range(half_size):
            if sequence[k] != sequence[l]:
                break
            if j == half_size - 1:
                return True
            k = (k - 1) if k > 0 else len(sequence) - 1
            l = (l + 1) % len(sequence)
    return False

#symmetrical_sequence = get_symmetrical_sequence(100)
#print("symmetrical:\t" + symmetrical_sequence)
#noisy_sequence = add_noise(symmetrical_sequence, 100)
#print("noisy:\t\t" + noisy_sequence)
#exit()

print("nb_bits = ", end="")
nb_bits = int(input())
noise = int(nb_bits * NOISE)
if MODE == "noisy":
    print("noise =", noise)
print("nb_examples = ", end="")
nb_examples = int(input())

pathlib.Path("datasets").mkdir(exist_ok=True)
file_name = "datasets/symmetry_" + str(nb_bits) + "_" + str(nb_examples) + ".csv"
with open(file_name, "w", encoding="utf-8") as file:
    for i in range(nb_examples):
        if random.randint(0, 1) == 0:
            if MODE == "trap":
                symmetrical_sequence = get_symmetrical_sequence(nb_bits)
                example = get_trap_sequence(nb_bits, symmetrical_sequence)
            else:
                example = get_random_sequence(nb_bits)
            example += ",0"
        else:
            if MODE == "noisy":
                symmetrical_sequence = get_symmetrical_sequence(nb_bits)
                example = add_noise(symmetrical_sequence, noise)
            elif MODE == "all_random":
                example = get_random_sequence(nb_bits)
            else:
                example = get_symmetrical_sequence(nb_bits)
            example += ",1"
        file.write(example + "\n")
