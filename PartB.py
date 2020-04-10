import io
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

alex_index = 499
req_score = 5

shows = np.loadtxt('Assignment-2-data/shows.txt', usecols=range(1), dtype='str', delimiter='\n')

shows_size = np.size(shows, 0)

user_shows = np.loadtxt('Assignment-2-data/user-shows.txt', usecols=range(shows_size))
user_size = np.size(user_shows, 0)

S = np.zeros(shows_size)

for t in range(0, req_score):
    print(t)
    current_show = user_shows[0:user_size, t]
    current_show_np = np.array([current_show])
    for j in range(req_score, shows_size):
        other_show = user_shows[0:user_size, j]
        other_user_np = np.array([other_show])
        S[t] += np.dot(cosine_similarity(other_user_np, current_show_np), current_show[t])

norm = np.linalg.norm(S)
normal_S = S/norm
# print(normal_S)

norm_S_order = np.argsort(normal_S)
for i in range(0, req_score):
    print(shows[norm_S_order[i]])