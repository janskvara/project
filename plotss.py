import matplotlib.pyplot as plt
import numpy as np

x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
random_data = [45,41,41,85,54,50,50,41,42,41,42,57,42,47,41,78,72,41,63,59]
resnet18  = [454,416,285,161,418,95,420,184,635,90,177,208,211,102,157,453,214,51,493,481]
bolt  = [175,398,399,273,172,172,434,151,539,250,84,311,186,280,415,517,434,575,496,435]
resnet152  = [70,41,42,49,41,50,47,52,52,41,41,41,41,41,41,41,41,41,41,41]

random_avg = np.average(random_data)
random_avg_full = np.full((20,1),random_avg)

resnet18_avg = np.average(resnet18)
resnet18_avg_full = np.full((20,1),resnet18_avg)

bolt_avg = np.average(bolt)
bolt_avg_full = np.full((20,1),bolt_avg)

d152_avg = np.average(resnet152)
d152_avg_full = np.full((20,1),d152_avg)

plt.figure()
plt.grid(linewidth=0.2)
plt.plot(x, random_data, color='red', alpha=1, linewidth=0, marker='o', markersize=4)
plt.plot(x, resnet18, color='orange', alpha=1, linewidth=0, marker='o', markersize=4)
plt.plot(x, bolt, color='blue', alpha=1, linewidth=0, marker='o', markersize=4)
plt.plot(x, resnet152, color='green', alpha=1, linewidth=0, marker='o', markersize=4)

plt.plot(x, random_avg_full, color='red', alpha=1, linewidth=0.9)
plt.plot(x, resnet18_avg_full, color='orange', alpha=1, linewidth=0.9)
plt.plot(x, bolt_avg_full, color='blue', alpha=1, linewidth=0.9)
plt.plot(x, d152_avg_full, color='green', alpha=1, linewidth=0.9)

plt.xlabel('Episode')
plt.ylabel('Achieved ingame score')
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
plt.yticks([0,50,100,150,200,250,300,350,400,450,500,550,600])
plt.legend(['ResNet-18 driven by random input',
            'ResNet-18',
            'ResNet-34',
            'ResNet-152',
'Avg score','Avg score','Avg score','Avg score'], bbox_to_anchor=(0,1.02,0.85,0.7), loc="lower left", mode="expand", borderaxespad=0, ncol=2)
plt.show()