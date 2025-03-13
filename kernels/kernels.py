import numpy as np
from itertools import product
from tqdm import tqdm

# --------------------------
# Spectrum Kernel
# --------------------------
def get_all_kmers(k):
    return [''.join(p) for p in product("ACGT", repeat=k)] 

def compute_kmer_counts(seq, k, kmers):
    counts = {kmer: 0 for kmer in kmers}
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if kmer in counts:
            counts[kmer] += 1
    # Return counts in the order of kmers
    return np.array([counts[kmer] for kmer in kmers])

class SpectrumKernel:
    def __init__(self, k=3):
        self.k = k
        self.kmers = get_all_kmers(k)
    
    def featurize(self, seqs):
        """
        Given a list or Series of DNA sequences, return a matrix where each row is the k-mer count vector.
        """
        return np.array([compute_kmer_counts(seq, self.k, self.kmers) for seq in seqs])
    
    def compute_kernel(self, seqs1, seqs2):
        """
        Compute the kernel matrix between two lists/Series of sequences.
        """
        X1 = self.featurize(seqs1)
        X2 = self.featurize(seqs2)
        return np.dot(X1, X2.T)

# --------------------------
# Missmatch Kernel
# --------------------------
class MismatchKernel:
    def __init__(self, k, m, kmers_dict, neighbors, alpha):
        self.k = k
        self.m = m
        self.kmers_dict = kmers_dict    # a dictionnary {kmer : idx}
        self.neighbors = neighbors      # a dictionnary {kmer : [neighb1, ...]}
        self.alpha = alpha

    def hamming_distance(self, s1, s2):
        """Compute the Hamming distance between two strings of equal length."""
        return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))

    def featurize(self, seqs):
        """
        Args:
            seqs (list or pandas.Series): A list or Series of DNA sequences.
        
        Returns:
            np.ndarray: A matrix of shape (num_sequences, 4^k) with k-mer counts.
        """
        n = len(seqs)
        all_kmers = list(self.kmers_dict.keys())
        d = len(all_kmers)
        X = np.zeros((n, d))
        # Process each sequence separately.
        for i, seq in enumerate(seqs):
            for j in range(len(seq) - self.k + 1):
                kmer = seq[j:j+self.k]
                # For this k-mer, update counts for all neighbors.
                for neighbor in self.neighbors[kmer]:
                    # Since every neighbor is a k-mer from "ACGT"^k, it is in self.kmer_to_index.
                    idx = self.kmers_dict[neighbor]
                    X[i, idx] += self.alpha ** self.hamming_distance(kmer, neighbor)
        return X

    def compute_kernel(self, seqs1, seqs2):
        """
        Args:
            seqs1 (list or pandas.Series): First set of sequences.
            seqs2 (list or pandas.Series): Second set of sequences.
        
        Returns:
            np.ndarray: The kernel matrix of shape (len(seqs1), len(seqs2)).
        """
        X1 = self.featurize(seqs1)
        X2 = self.featurize(seqs2)
        return np.dot(X1, X2.T)

# --------------------------
# Local Alignment Kernel
# --------------------------
class LocalAlignmentKernel:
    def __init__(self, beta=1.0, match_score=1.0, mismatch_score=-1.0):
        """
        Args:
            beta (float): Scaling parameter used in the exponential weighting.
            match_score (float): Score for matching nucleotides.
            mismatch_score (float): Score for mismatching nucleotides.
        """
        self.beta = beta
        self.match_score = match_score
        self.mismatch_score = mismatch_score

    def substitution_score(self, a, b):
        """
        Args:
            a (str): First character.
            b (str): Second character.
        
        Returns:
            float: match_score if a equals b, else mismatch_score.
        """
        return self.match_score if a == b else self.mismatch_score

    def compute_alignment(self, x, y):
        """
        Compute the local alignment kernel between two sequences x and y.
        
        This dynamic programming routine sums over all contiguous (gap-free) 
        alignments. For each cell (i,j) in the DP matrix L, we set
        
            L[i,j] = exp(beta * s(x[i-1], y[j-1])) * (1 + L[i-1, j-1])
        
        and the final kernel value is the sum of all entries in L.
        
        Args:
            x (str): First DNA sequence.
            y (str): Second DNA sequence.
            
        Returns:
            float: Kernel value between x and y.
        """
        n = len(x)
        m = len(y)
        # DP matrix with an extra row and column (initialized to zero)
        L = np.zeros((n + 1, m + 1))
        
        # Fill in the DP matrix using our recurrence
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                score = self.substitution_score(x[i - 1], y[j - 1])
                L[i, j] = np.exp(self.beta * score) * (1 + L[i - 1, j - 1])
        
        # The kernel value is the sum of contributions from all local alignments
        return L.sum()

    def compute_kernel(self, seqs1, seqs2):
        """
        Args:
            seqs1 (list or pandas.Series): First set of DNA sequences.
            seqs2 (list or pandas.Series): Second set of DNA sequences.
            
        Returns:
            np.ndarray: A kernel matrix of shape (len(seqs1), len(seqs2)).
        """
        n1 = len(seqs1)
        n2 = len(seqs2)
        seqs1 = list(seqs1)
        seqs2 = list(seqs2)
        K = np.zeros((n1, n2))
        for i in tqdm(range(n1)):
            for j in range(n2):
                K[i, j] = self.compute_alignment(seqs1[i], seqs2[j])
        return K

    def featurize(self, seqs):
        """
        Args:
            seqs (list or pandas.Series): A list or Series of DNA sequences.
        
        Returns:
            np.ndarray: A 1D array where each entry is the self-kernel value of a sequence.
        """
        return np.array([self.compute_alignment(seq, seq) for seq in seqs])

# --------------------------
# Weighted Degree Kernel
# --------------------------
class WeightedDegreeKernel:
    def __init__(self, D=3, weights=None):
        """
        Args:
            D (int): Maximum substring length (degree). 
                     It is assumed that 1 <= d <= D.
            weights (list of float, optional): A list of weights [β1, β2, ..., β_D].
                     If None, uniform weights of 1.0 are used.
        """
        self.D = D
        if weights is None:
            self.weights = [1.0] * D
        else:
            if len(weights) != D:
                raise ValueError("Length of weights must be equal to D.")
            self.weights = weights

    def _kernel_between(self, x, y):
        """
        Compute the weighted degree kernel value between two aligned sequences.
        
        The computation sums over substring lengths d = 1 to D and, for each d,
        over all positions i where a substring of length d can be extracted.
        
        Args:
            x (str): First DNA sequence.
            y (str): Second DNA sequence.
        
        Returns:
            float: Kernel value between x and y.
        """
        if len(x) != len(y):
            raise ValueError("Sequences must have the same length for the weighted degree kernel.")
        
        L = len(x)
        K_val = 0.0
        # For each degree (substring length) d.
        for d in range(1, self.D + 1):
            weight = self.weights[d - 1]
            # For each valid starting position i.
            for i in range(L - d + 1):
                if x[i:i+d] == y[i:i+d]:
                    K_val += weight
        return K_val

    def compute_kernel(self, seqs1, seqs2):
        """
        Args:
            seqs1 (list or pandas.Series): First set of DNA sequences.
            seqs2 (list or pandas.Series): Second set of DNA sequences.
        
        Returns:
            np.ndarray: A kernel matrix of shape (len(seqs1), len(seqs2)).
        """
        n1 = len(seqs1)
        n2 = len(seqs2)
        seqs1 = list(seqs1)
        seqs2 = list(seqs2)
        K_matrix = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                K_matrix[i, j] = self._kernel_between(seqs1[i], seqs2[j])
        return K_matrix

    def featurize(self, seqs):
        """
        Args:
            seqs (list or pandas.Series): A list or Series of DNA sequences.
        
        Returns:
            np.ndarray: A 1D array where each entry is the self-kernel value for a sequence.
        """
        return np.array([self._kernel_between(seq, seq) for seq in seqs])
