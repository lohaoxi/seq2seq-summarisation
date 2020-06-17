import collections
import numpy as np

# BLEU
# https://www.aclweb.org/anthology/P02-1040.pdf
# https://pytorch.org/text/data_metrics.html

def get_ngram(lst, n):
    ngram = [tuple(lst[i:i+n]) for i in range(len(lst) - n + 1)]
    return ngram

def ngram_iterator(lst, N=4):
    ngrams = []
    for n in range(N):
        ngram = get_ngram(lst, n + 1)
        for ww in ngram:
            ngrams.append(ww)
    return ngrams

def ngram_counter(lst, N=4):
    ngram_counts = collections.Counter()
    ngrams = ngram_iterator(lst, N)
    for ngram in ngrams:
        ngram_counts[ngram] =+ 1
    return ngram_counts

def divide(numerator, denominator):
    if np.min(denominator) > 0:
        return numerator / denominator
    else:
        return 0.0
    
def merge_ngram_counter(lsts, N):
    counter = collections.Counter()
    for lst in lsts:
        counter |= ngram_counter(lst, N)
    return counter

def update_ngram_counts(counts, counter):
    for ngram in counter:
        counts[len(ngram)-1] += counter[ngram]
    return counts
    
def BLEU(references_corpus, candidate_corpus, N=4, weights=[float(1/4)]*4):

    r = 0.0
    c = 0.0
    clip_counts = np.zeros(N)
    total_counts = np.zeros(N)
    
    for (refs, can) in zip(references_corpus, candidate_corpus):
        refs.sort(key=lambda x: abs(len(can) - len(x)))
        
        c += len(can)
        r += len(refs[0])
        
        refs_counter = merge_ngram_counter(refs, N)
        can_counter = ngram_counter(can, N)
        clip_counter = can_counter & refs_counter
        
        clip_counts = update_ngram_counts(clip_counts, clip_counter)
        total_counts = update_ngram_counts(total_counts, can_counter)

    pn = divide(clip_counts, total_counts)
    BP = np.exp(min(1 - (r/c), 0))
    bleu_score = BP * np.exp(np.dot(weights, np.log(pn)))
        
    return(bleu_score)


candidate_corpus = [['My', 'full', 'pytorch', 'test'], 
                    ['Another', 'Sentence']]
references_corpus = [[['My', 'full', 'pytorch', 'test'], ['Completely', 'Different']], 
                     [['No', 'Match']]]
# BLEU(references_corpus, candidate_corpus)


# ROUGE
# https://www.aclweb.org/anthology/W04-1013.pdf

def match(ref, sys):
    intersection = set(ref) & set(sys)
    return len(intersection)

def single_rouge_n(ref, sys, N):
    # 1 to 1 rouge score
    ref_iterator = get_ngram(ref, N)
    sys_iterator = get_ngram(sys, N)
    match_count = match(ref_iterator, sys_iterator)
    rec_count = len(ref_iterator)
    
    return match_count, rec_count
    
def multi_rouge_n(refs, sys, N):
    # 2.1 Multiple references
    temp_rouge = np.zeros([len(refs), 2])
    for i in range(len(refs)):
        temp_rouge[i, :] = single_rouge_n(refs[i], sys, N)
    rogue = divide(temp_rouge[:, 0], temp_rouge[:, 1])
    ind = np.argmax(rogue)
    
    match_count, rec_count = temp_rouge[ind, ]
    return match_count, rec_count
    

def ROGUE_N(reference_summaries, system_summaries, N):
    
    match_counts = 0.0
    rec_counts = 0.0
    
    for (refs, sys) in zip(reference_summaries, system_summaries):
        match_count, rec_count = multi_rouge_n(refs, sys, N)
        match_counts += match_count
        rec_counts += rec_count
        
    rouge_n = divide(match_counts, rec_counts)
    return rouge_n

system_summaries = ['the cat was found under the bed'.split()]
reference_summaries = [['the cat was under the bed'.split()]]
# ROGUE_N(reference_summaries, system_summaries, 2)


def lcs_table(X, Y):
    # https://www.tutorialspoint.com/design_and_analysis_of_algorithms/design_and_analysis_of_algorithms_longest_common_subsequence.htm
    m = len(X)
    n = len(Y)
    C = np.zeros([m + 1, n + 1])
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                C[i, j] = C[i - 1, j - 1] + 1 
            else:
                if C[i - 1, j] > C[i, j - 1]:
                    C[i, j] = C[i - 1, j]
                else:
                    C[i, j] = C[i, j - 1]
    return C

def lcs(X, Y):
    
    C = lcs_table(X, Y)
    i = len(X)
    j = len(Y)
    
    def lcs_path(i, j):
        if i == 0 or j == 0:
            return []
        elif X[i - 1] == Y[j - 1]:
            return lcs_path(i - 1, j - 1) + [X[i - 1]]
        
        elif C[i - 1, j] > C[i, j - 1]:
            return lcs_path(i - 1, j)
        else:
            return lcs_path(i, j - 1)
            
    return lcs_path(i, j)

def ROUGE_L_sentence(reference_sentence, system_sentence):
    
    LCS = len(lcs(reference_sentence, system_sentence))
    m = len(reference_sentence)
    n = len(system_sentence)
    R_lcs = divide(LCS, m)
    P_lcs = divide(LCS, n)
    beta = divide(P_lcs, R_lcs)
    F_lcs = divide((1 + beta ** 2) * R_lcs * P_lcs, R_lcs + (beta ** 2) * P_lcs)
    
    return F_lcs

S1 = 'police killed the gunman'.split()
S2 = 'police kill the gunman'.split()
S3 = 'the gunman kill police'.split()
# ROGUE_N([[S1]], [S2], 2) == 1/3
# ROGUE_N([[S1]], [S3], 2) == 1/3
# ROUGE_L_sentence(S1, S2) == 0.75
# ROUGE_L_sentence(S1, S3) == 0.5

def lcs_u(X, Ys):
    
    lcs_union = [lcs(X, Y) for Y in Ys]
    lcs_union_len = sum([len(ss) for ss in lcs_union])
    lcs_union = set([ww for ss in lcs_union for ww in ss])
        
    return divide(len(lcs_union), lcs_union_len)

ri = 'w1 w2 w3 w4 w5'.split()
c1 = 'w1 w2 w6 w7 w8'.split()
c2 = 'w1 w3 w8 w9 w5'.split()
# lcs_u(ri, [c1, c2]) == 0.8

def ROUGE_L_summary(reference_sentences, candidate_sentences):

    m = sum([len(ref) for ref in reference_sentences])
    n = sum([len(can) for can in candidate_sentences])
    
    sum_LCS_u = sum([lcs_u(candidate_sentences, ref) for ref in reference_sentences])
    R_lcs = divide(sum_LCS_u, m)
    P_lcs = divide(sum_LCS_u, n)
    beta = divide(P_lcs, R_lcs)
    F_lcs = divide((1 + beta ** 2) * R_lcs * P_lcs, R_lcs + (beta ** 2) * P_lcs)

    return F_lcs




def f(k, alpha):
    # weighting function
    return k ** alpha

def inverse_f(k, alpha):
    return k ** (1/alpha)

def wlcs(X, Y, alpha):
    m = len(X)
    n = len(Y)
    c = np.zeros([m + 1, n + 1])
    w = np.zeros([m + 1, n + 1])
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                k = w[i - 1, j - 1]
                c[i, j] = c[i - 1, j - 1] + f(k + 1, alpha) - f(k, alpha)
                w[i, j] = k + 1
            else:
                if c[i - 1, j] > c[i, j - 1]:
                    c[i, j] = c[i - 1, j]
                else:
                    c[i, j] = c[i, j - 1]
                w[i, j] = 0

    return c[m, n]


def ROUGE_W(reference_sentence, candidate_sentence, alpha=2):
    
    WLCS = wlcs(reference_sentence, candidate_sentence, alpha)
    m = len(reference_sentence)
    n = len(candidate_sentence)
    
    R_wlcs = inverse_f(divide(WLCS, f(m, alpha)), alpha)
    P_wlcs = inverse_f(divide(WLCS, f(n, alpha)), alpha)
    beta = divide(P_wlcs, R_wlcs)
    F_wlcs = divide((1 + beta ** 2) * R_wlcs * P_wlcs, R_wlcs + (beta ** 2) * P_wlcs)
    
    return F_wlcs


X = 'A B C D E F G'.split()
Y1 = 'A B C D H I K'.split()
Y2 = 'A H B K C I D'.split()
# ROUGE_W(X, Y1) 
# ROUGE_W(X, Y2)

def combin(n, r):
    return int(np.math.factorial(n) / (np.math.factorial(n - r) * np.math.factorial(r)))

def skip_bigram(lst):
    skip_grams = [(lst[i], lst[j]) for i in range(len(lst)) for j in range(len(lst)) if i < j]
    return skip_grams

def skip2(X, Y):
    return match(skip_bigram(X), skip_bigram(Y))

def ROUGE_S(reference_sentence, candidate_sentence):
    
    m = len(reference_sentence)
    n = len(candidate_sentence)
    SKIP2 = skip2(reference_sentence, candidate_sentence)
    
    R_skip2 = divide(SKIP2, combin(m, 2))
    P_skip2 = divide(SKIP2, combin(n, 2))
    beta = divide(P_skip2, R_skip2)
    F_skip2 = divide((1 + beta ** 2) * R_skip2 * P_skip2, R_skip2 + (beta ** 2) * P_skip2)
    
    return F_skip2

def ROUGE_SU(reference_sentence, candidate_sentence):
    return ROUGE_S(['<sos>'] + reference_sentence, ['<sos>'] + candidate_sentence)

    

S1 = 'police killed the gunman'.split()
S2 = 'police kill the gunman'.split()
S3 = 'the gunman kill police'.split()
S4 = 'the gunman police killed'.split()
S5 = 'gunman the killed police'.split()

# skip_bigram(S1)
# ROUGE_S(S1, S2)
# ROUGE_S(S1, S3)
# ROUGE_S(S1, S4)