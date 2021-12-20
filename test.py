import pickle


with open('dataset\model.pkl', 'rb') as f:
    data = pickle.load(f)

print('boom')