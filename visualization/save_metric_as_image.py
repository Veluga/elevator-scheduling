from sys import stdin, argv
import matplotlib.pyplot as plt
import json

x = []
y = []
for line in stdin:
    try:
        datapoint = json.loads(line)
        x.append(datapoint["step"])
        y.append(datapoint["value"])
    except:
        pass

plt.plot(x, y, "r-")
plt.xlabel("Timestep")
plt.ylabel(argv[2])
plt.savefig("{}/plots/{}.png".format(argv[1], argv[2]))