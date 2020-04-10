import io
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

alex_index = 499
req_score = 5
K = 10

shows = np.loadtxt('Assignment-2-data/shows.txt', usecols=range(1), dtype='str', delimiter='\n')
shows_size = np.size(shows, 0)

user_shows = np.loadtxt('Assignment-2-data/user-shows.txt', usecols=range(shows_size))
user_size = np.size(user_shows, 0)

u, s, vh = np.linalg.svd(user_shows, full_matrices=True)

# print(u, s, vh)

Rstar = np.zeros((len(u), len(vh)))
for i in range(0, K):
    Rstar += s[i] * np.outer(u.T[i], vh[i])

r_alex = Rstar[alex_index, 0:shows_size]
r_alex_req = r_alex[0:req_score]

norm_S_order = np.argsort(r_alex_req)
for i in range(0, req_score):
    print(shows[norm_S_order[i]])