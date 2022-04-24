import numpy as np
import matplotlib.pyplot as plt
Fs=100        # 采样率为100Hz,信号时长t=10s
Size=1000     #采样点数=采样率*信号时长=100*10=1000
t=np.arange(0,Size)/Fs
x1=np.sin(2*np.pi*1*t)


#*******绘图部分********#
fig=plt.figure()
ax1= fig.add_subplot(121)
plt.plot(t,x1)
plt.title('without Spindle ')

#隐藏边框线
ax1=plt.gca()


ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)
#plt.xticks([])#隐藏x轴刻度
#plt.yticks([])#隐藏y轴刻度

ax1= fig.add_subplot(122)
plt.plot(t,x1)
plt.title('with Spindle ',c='red')
ax1.spines['bottom'].set_linewidth('2.0')#设置边框线宽为2.0
ax1.spines['bottom'].set_color('red')

plt.show()
