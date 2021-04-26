from sys import stdin, argv
import matplotlib.pyplot as plt
import json

"""Show matplotlib plot for single metric (e.g. avg_return).
Used by visualize_metric script.
"""

x = []
y = []
for line in stdin:
    datapoint = json.loads(line)
    x.append(datapoint["step"])
    y.append(datapoint["value"])

plt.plot(x, y, "r-")
plt.xlabel("Timestep")
plt.ylabel(argv[1])
plt.show()