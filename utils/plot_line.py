from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']

x = [16, 24, 32, 48]
y1 = [36.6, 34.5, 22.4, 22.4]
y2 = [36.7, 33.2, 22.3, 22.4]

plt.figure(figsize=(10, 6))
plt.plot(x, y1, marker='o', linestyle='-',color='blue', label='')

