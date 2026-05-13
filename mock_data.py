import pandas as pd
import random

titles = ['Movie ' + str(i) for i in range(1, 31)]
data = {
    'id': list(range(1, 31)),
    'title': titles,
    'overview': ['Overview for ' + t for t in titles],
    'genres': ["[{'name': 'Action'}]" for _ in range(30)],
    'keywords': ["[{'name': 'test'}]" for _ in range(30)],
    'poster_path': ['/test.jpg' for _ in range(30)],
    'popularity': [random.uniform(50, 100) for _ in range(30)]
}

df = pd.DataFrame(data)

real_data = {
    'id': [101, 102, 103, 104, 105],
    'title': ['The Dark Knight', 'Batman Begins', 'Toy Story', 'Toy Story 2', 'The Matrix'],
    'overview': ['Batman fights Joker', 'Batman origin', 'Toys alive', 'More toys', 'Neo wakes up'],
    'genres': ["[{'name': 'Action'}]", "[{'name': 'Action'}]", "[{'name': 'Family'}]", "[{'name': 'Family'}]", "[{'name': 'Sci-Fi'}]"],
    'keywords': ["[{'name': 'batman'}]", "[{'name': 'batman'}]", "[{'name': 'toys'}]", "[{'name': 'toys'}]", "[{'name': 'matrix'}]"],
    'poster_path': ['/dk.jpg', '/bb.jpg', '/ts.jpg', '/ts2.jpg', '/tm.jpg'],
    'popularity': [100, 95, 90, 85, 99]
}
df_real = pd.DataFrame(real_data)
final_df = pd.concat([df, df_real], ignore_index=True)
final_df.to_csv('recommender/data/mock.csv', index=False)
print('Generated 35 row dataset!')
