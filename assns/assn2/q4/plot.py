import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
f1 = "gd_data"
f2 = "scd_data"

gd = pd.read_csv(f1)
scd = pd.read_csv(f2)
obj_gd = gd['obj_val_gd']
obj_scd = scd['obj_val_scd']
a = np.ndarray(len(obj_gd))
for i in range(len(obj_gd)):
	a[i] = i*300000*54*100
b = np.ndarray(len(obj_scd))
for i in range(len(obj_scd)):
        b[i] = i*100000*54
obj_scd = scd['obj_val_scd']
plt.plot(gd['time'],gd['obj_val_gd'],alpha = 0.5)
plt.plot(scd['time'],scd['obj_val_scd'],alpha = 0.5)
plt.xlabel("Elapsed Theoretical Time")
plt.ylabel("Objective value")
#plt.xlim(-50000000,3000000000)
plt.legend()
fig = plt.gcf()
plt.show()
