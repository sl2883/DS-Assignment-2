import io
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

alex_index = 499
req_score = 5

shows = np.loadtxt('Assignment-2-data/shows.txt', usecols=range(1), dtype='str', delimiter='\n')

shows_size = np.size(shows, 0)
user_shows = np.loadtxt('Assignment-2-data/user-shows.txt', usecols=range(shows_size))

r_alex = user_shows[alex_index, 0:shows_size]
r_alex_np = np.array([r_alex])

S = np.zeros(shows_size)

for t in range(0, req_score):
    print(t)
    for j in range(0, np.size(user_shows, 0)):
        other_user = user_shows[j, 0:shows_size]
        other_user_np = np.array([other_user])
        S[t] += np.dot(cosine_similarity(other_user_np, r_alex_np), other_user[t])

norm = np.linalg.norm(S)
normal_S = S/norm
# print(normal_S)

norm_S_order = np.argsort(normal_S)
for i in range(0, req_score):
    print(shows[norm_S_order[i]])

