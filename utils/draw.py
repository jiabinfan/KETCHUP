import matplotlib.pyplot as plt

# Data from the table
x = [1, 2, 3, 4, 5, 6, 7]  # N-step
y1 = [13.77, 13.17, 11.46, 9.99, 9.05, 8.09, 7.85]  # 1-20 ATL(%)
y2 = [60.71, 60.5, 60.51, 60.51, 59.97, 59.83, 59.8]  # 21-30 ATL(%)
y3 = [22.21, 22.92, 24.48, 26.10, 27.47, 28.37, 28.58]  # 31-40 ATL(%)
y4 = [3.31, 3.41, 3.55, 3.6, 3.51, 3.71, 3.77]  # 40+ ATL(%)

# Plotting the lines
plt.plot(x, y1, label='1-20 ATL(%)', marker='o')
plt.plot(x, y2, label='21-30 ATL(%)', marker='o')
plt.plot(x, y3, label='31-40 ATL(%)', marker='o')
plt.plot(x, y4, label='40+ ATL(%)', marker='o')

# Adding labels and title
plt.xlabel('N-step')
plt.ylabel('Percentage (%)')
plt.title('Percentage across N-steps')

# Adding a legend
plt.legend()
plt.savefig('percentage_across_n_steps.png')
# Displaying the plot
plt.show()
