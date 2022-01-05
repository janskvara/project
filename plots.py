import matplotlib.pyplot as plt
import numpy as np

x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
random_data = [45,41,41,85,54,50,50,41,42,41,42,57,42,47,41,78,72,41,63,59]
data_430 = [92,162,133,152,87,189,143,76,50,69,118,109,62,97,217,56,161,99,89,229]
data_bolt_34 = [175,398,399,273,172,172,434,151,539,250,84,311,186,280,415,517,434,575,496,435]
emil_196 = []
jano_bot = [2100]

random_avg = np.average(random_data)
random_avg_full = np.full((20,1),random_avg)

d430_avg = np.average(data_430)
d430_avg_full = np.full((20,1),d430_avg)

bolt_avg = np.average(data_bolt_34)
bolt_avg_full = np.full((20,1),bolt_avg)

emil_avg = np.average(emil_196)
emil_avg_full = np.full((20,1),emil_avg)

plt.figure()
plt.grid(linewidth=0.2)
plt.plot(x, random_data, color='red', alpha=1, linewidth=0, marker='o', markersize=5)
plt.plot(x, random_avg_full, color='red', alpha=1, linewidth=0.8)

plt.plot(x, data_430, color='blue', alpha=1, linewidth=0, marker='o', markersize=5)
plt.plot(x, d430_avg_full, color='blue', alpha=1, linewidth=0.8)

plt.plot(x, data_bolt_34, color='green', alpha=1, linewidth=0, marker='o', markersize=5)
plt.plot(x, bolt_avg_full, color='green', alpha=1, linewidth=0.8)

#plt.plot(x, emil_196, color='orange', alpha=1, linewidth=0, marker='o', markersize=5)
#plt.plot(x, emil_avg_full, color='orange', alpha=1, linewidth=0.8)


plt.xlabel('Episode')
plt.ylabel('Achieved ingame score')
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
plt.yticks([0,50,100,150,200,250,300,350,400,450,500,550,600])
plt.legend(['Model driven by random input','Learning dataset of 80000 screenshots','ResNet34 on 196000 screenshots','Learning dataset of 196000 screenshots'], bbox_to_anchor=(0,1.02,0.7,0.5), loc="lower left", mode="expand", borderaxespad=0, ncol=1)
plt.show()