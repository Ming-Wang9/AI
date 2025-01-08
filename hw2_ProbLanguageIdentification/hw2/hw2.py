import sys
import math
import string


def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)

def shred(filename):
    #Using a dictionary here. You may change this to any data structure of
    #your choice such as lists (X=[]) etc. for the assignment

    # took suggestion from the link below
    # https://stackoverflow.com/questions/453576/is-there-a-fast-way-to-generate-a-dict-of-the-alphabet-in-python
    X = dict(zip(string.ascii_uppercase, [0]*26))
    with open (filename,encoding='utf-8') as f:
        # TODO: add your code here
        f_content = f.read()
        for char in f_content:
            char = char.upper()  # Move this inside the loop
            if char in X:
                X[char] += 1

    print("Q1")
    for letter in string.ascii_uppercase:
        print(f"{letter} {X[letter]}")

    return X

# TODO: add your code here for the assignment
# You are free to implement it as you wish!
# Happy Coding!
def q2(X, e, s):
    x1 = X['A']  
    e1 = e[0]    
    s1 = s[0]   
    
    if x1 > 0:
        q2_e = x1 * math.log(e1)
    else:
        q2_e = -0.0000 

    if x1 > 0:
        q2_s = x1 * math.log(s1)
    else:
        q2_s = -0.0000 
    
    print("Q2")
    print(f"{q2_e:.4f}")
    print(f"{q2_s:.4f}")

def q3(X, prior_english, prior_spanish, e, s):
    F_english = log_likelihood(X, prior_english, e)
    F_spanish = log_likelihood(X, prior_spanish, s)
    
    print("Q3")
    print(f"{F_english:.4f}")
    print(f"{F_spanish:.4f}")
    
    return F_english, F_spanish

def q4(F_english, F_spanish):
    diff = F_spanish - F_english
    
    if diff >= 100:
        p_english = 0.0
    elif diff <= -100:
        p_english = 1.0
    else:
        p_english = 1 / (1 + math.exp(diff))
    
    print("Q4")
    print(f"{p_english:.4f}")


def log_likelihood (X, priro_prb, vector):
    log = math.log(priro_prb)
    for i, key_value in enumerate(X.values()):
        if key_value > 0:
            log += key_value * math.log(vector[i])
    return log


def language_identification (filename, prior_prb_e, prior_pro_s):
    X = shred(filename)
    e,s = get_parameter_vectors()
    q2(X, e, s)

    F_english = log_likelihood(X, prior_prb_e, e)
    F_spanish = log_likelihood(X,  prior_pro_s, s)
    q3(X, prior_prb_e, prior_pro_s, e, s)
    q4(F_english, F_spanish)

    comparison = F_spanish - F_english
    if comparison >= 100:
        return 0
    elif comparison <= -100:
        return 1
    else:
        return 1 / (1 + math.exp(comparison))

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 hw2.py [letter_file] [prior_english] [prior_spanish]")
        sys.exit(1)

    letter_file = sys.argv[1]
    prior_english = float(sys.argv[2])
    prior_spanish = float(sys.argv[3])

    language_identification(letter_file, prior_english, prior_spanish)