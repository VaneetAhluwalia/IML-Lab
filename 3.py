import matplotlib.pyplot as plt

m = float(input("Enter value of m : "))
c = float(input("Enter value of c : "))
x_val = []
y_val = []

for i in range(5):
    x = float(input(f"Enter x coordinate {i+1} : "))
    y = float(input(f"Enter y coordinate {i+1} : "))
    x_val.append(x)
    y_val.append(y)

errors = []
y_pred = []
totalerror = 0 

for i in range(5):
    y_pre = m*x_val[i] + c
    y_pred.append(y_pre)
    error = abs(y_pre - y_val[i])
    errors.append(error)
    totalerror += error

print("total error : ", totalerror)

# Plot actual points
plt.scatter(x_val, y_val, color='blue', label="Actual Points")

# Plot chosen line
x_line = [min(x_val)-1, max(x_val)+1]
y_line = [m*x + c for x in x_line]
plt.plot(x_line, y_line, color='black', label="Chosen Line")

# Draw perpendiculars
colors = ['red', 'green', 'purple', 'orange', 'brown']

for i in range(5):
    x0, y0 = x_val[i], y_val[i]
    
    # foot of perpendicular formula
    xf = (x0 + m*(y0 - c)) / (1 + m**2)
    yf = m*xf + c
    
    plt.plot([x0, xf], [y0, yf], color=colors[i], linestyle='--')

plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Error Visualization using Perpendicular Distances")
plt.legend()
plt.show()
