from system_simulation import SystemDynamics
from pid_controller import PIDController
from ziegler_nichols import * 
from plotting import plot_PID
import matplotlib.pyplot as plt

num = [20]
den = [1, 32, 140, 0]
system = SystemDynamics(num, den)

kpu = 224  # Ultimate gain, obtained via Root Locus Calculation

t, y = find_kpu_response(system, kpu)

#plot the results   
plt.plot(t, y)
plt.title('PID Controller Tuning')
plt.xlabel('Time')
plt.ylabel('Response')
plt.grid(True)
plt.show()
