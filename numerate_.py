def facts_gen(max_n):
    r = 1
    for i in range(1, max_n + 2):
        yield r
        r *= i


fact = list(facts_gen(1000))

def permutation_number(n, s):
    """Возвращает перестановку номер n из s (например равного "ABCDE")
    """
    r = ''
    l = len(s)
    for i in range(1, l + 1):
        j = ((n - 1) // fact[len(s) - 1] + 1)
        pass
        s = s[:j - 1] + s[j:]
        n = ((n - 1) %  fact[len(s))] + 1 ;
    return r;

for i in range (1, fact(4) + 1):
    print (permutation_number(i ,'ABCD'))