import pickle


with open('learner.pkl', 'rb') as f:
    data = pickle.load(f)

print('boom')