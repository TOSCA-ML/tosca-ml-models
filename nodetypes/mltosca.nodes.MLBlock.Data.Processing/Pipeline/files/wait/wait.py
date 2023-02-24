import time
import pickle
import os.path


while not os.path.exists('/home/artjom/mltosca/df.pkl'):
  time.sleep(1)

with open(path, 'rb') as file:
  model = pickle.load(file)

# open("/home/artjom/Downloads/order.txt", "w")
