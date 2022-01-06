import matplotlib.pyplot as plt
import numpy as np

x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
random_data = [45,41,41,85,54,50,50,41,42,41,42,57,42,47,41,78,72,41,63,59]
densenet_5_196  = [50,41,42,61,51,64,42,58,48,53,49,50,48,51,49,42,49,42,42,42]
alex_5_196  = [45,159,356,179,364,362,183,144,159,50,564,163,183,103,70,163,192,56,67,177]
squeez_5_196  = [437,439,181,205,41,240,42,423,386,201,183,49,461,145,361,66,260,301,334,190]
bolt_196_34 = [454,416,285,161,418,95,420,184,635,90,177,208,211,102,157,453,214,51,493,481]
vgg_196 = [92,162,133,152,87,189,143,76,50,69,118,109,62,97,217,56,161,99,89,229]

random_avg = np.average(random_data)
random_avg_full = np.full((20,1),random_avg)

d30_avg = np.average(densenet_5_196)
d30_avg_full = np.full((20,1),d30_avg)

d80_avg = np.average(alex_5_196)
d80_avg_full = np.full((20,1),d80_avg)

d150_avg = np.average(squeez_5_196)
d150_avg_full = np.full((20,1),d150_avg)

d430_avg = np.average(bolt_196_34)
d430_avg_full = np.full((20,1),d430_avg)

vgg_avg = np.average(vgg_196)
vgg_full = np.full((20,1),vgg_avg)

plt.figure()
plt.grid(linewidth=0.2)
plt.plot(x, random_data, color='red', alpha=1, linewidth=0, marker='o', markersize=4)
plt.plot(x, densenet_5_196, color='orange', alpha=1, linewidth=0, marker='o', markersize=4)
plt.plot(x, alex_5_196, color='blue', alpha=1, linewidth=0, marker='o', markersize=4)
plt.plot(x, squeez_5_196, color='green', alpha=1, linewidth=0, marker='o', markersize=4)
plt.plot(x, bolt_196_34, color='purple', alpha=1, linewidth=0, marker='o', markersize=4)
plt.plot(x, vgg_196, color='magenta', alpha=1, linewidth=0, marker='o', markersize=4)

plt.plot(x, random_avg_full, color='red', alpha=1, linewidth=0.9)
plt.plot(x, d30_avg_full, color='orange', alpha=1, linewidth=0.9)
plt.plot(x, d80_avg_full, color='blue', alpha=1, linewidth=0.9)
plt.plot(x, d150_avg_full, color='green', alpha=1, linewidth=0.9)
plt.plot(x, d430_avg_full, color='purple', alpha=1, linewidth=0.9)
plt.plot(x, vgg_full, color='magenta', alpha=1, linewidth=0.9)

plt.xlabel('Episode')
plt.ylabel('Achieved ingame score')
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
plt.yticks([0,50,100,150,200,250,300,350,400,450,500,550,600])
plt.legend(['ResNet-18 driven by random input',
            'ResNet-18 on 30 000 images',
            'ResNet-18 on 80 000 images',
            'ResNet-18 on 150 000 images',
            'ResNet-18 on 196 000 images',
            'ResNet-18 on 430 000 images',
'Avg score','Avg score','Avg score','Avg score','Avg score','Avg score'], bbox_to_anchor=(0,1.02,0.85,0.7), loc="lower left", mode="expand", borderaxespad=0, ncol=2)
plt.show()