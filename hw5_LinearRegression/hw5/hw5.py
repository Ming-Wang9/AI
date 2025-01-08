import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

# Function to load data from the CSV file
def load_data(filename):
    years = []
    frozen_days = []
    with open(filename) as csvFile:
        reader = csv.DictReader(csvFile)
        for row in reader:
            years.append(int(row["year"][:4]))  # Use first 4 characters of year
            frozen_days.append(int(row["days"]))
    return np.array(years), np.array(frozen_days)

# Function to visualize the data (Q2)
def visualize_data(years, frozen_days):
    plt.plot(years, frozen_days, marker='o')
    plt.xlabel("Year")
    plt.ylabel("Number of Frozen Days")
    plt.title("Year vs Number of Frozen Days")
    plt.grid(True)
    plt.savefig("data_plot.jpg")  # Save the plot as "data_plot.jpg"

# Min-max normalization (Q3)
def normalize_data(years):
    min_year = np.min(years)
    max_year = np.max(years)
    normalized_years = (years - min_year) / (max_year - min_year)
    X_normalized = np.column_stack((normalized_years, np.ones(len(years))))  # Augment with 1s
    print("Q3:")
    print(X_normalized)
    return X_normalized, min_year, max_year

# Closed-form solution for linear regression (Q4)
def closed_form_solution(X_normalized, frozen_days):
    weights = np.linalg.inv(X_normalized.T @ X_normalized) @ X_normalized.T @ frozen_days
    print("Q4:")
    print(weights)
    return weights

# Gradient descent for linear regression (Q5)
def gradient_descent(X_normalized, Y, learning_rate, iterations):
    m = len(Y)
    theta = np.zeros(2)  # Initialize theta (bias and weight)
    cost_history = []

    print("Q5a:")  # Indicate the start of the output for Q5a
    print(theta)  # Output the initial weights [0, 0]

    for i in range(iterations):
        predictions = X_normalized @ theta  # Calculate predictions
        errors = predictions - Y  # Calculate errors
        cost = (1 / (2 * m)) * np.sum(errors ** 2)  # Mean squared error
        cost_history.append(cost)  # Track cost for plotting

        # Update weights (theta)
        theta[0] -= (learning_rate / m) * np.sum(errors)  # Update bias
        theta[1] -= (learning_rate / m) * np.sum(errors * X_normalized[:, 0])  # Update weight

        # Print weights every 10 iterations
        if i % 10 == 0 and i < iterations - 1:
            print(theta)  # Output the current weights

    # Print the final weights to ensure proper convergence
    print(theta)

    # Print learning rate and iterations as Q5b and Q5c
    print("Q5b: " + str(learning_rate))
    print("Q5c: " + str(iterations))

    return theta, cost_history
    return theta, cost_history

# Prediction for 2023-24 (Q6)
def predict_for_2023(weights, min_year, max_year):
    x_2023 = (2023 - min_year) / (max_year - min_year)  # Normalize 2023
    y_2023 = weights[0] * x_2023 + weights[1]
    print("Q6: " + str(y_2023))
    return y_2023

# Model interpretation (Q7)
def model_interpretation(weights):
    if weights[0] > 0:
        symbol = ">"
    elif weights[0] < 0:
        symbol = "<"
    else:
        symbol = "="

    print("Q7a: " + symbol)
    if symbol == ">":
        print("Q7b: More frozen days over time.")
    elif symbol == "<":
        print("Q7b: Fewer frozen days over time.")
    else:
        print("Q7b: Frozen days remain constant.")

# Model limitation (Q8)
def model_limitation(weights, min_year, max_year):
    x_star = min_year + (-weights[1] / weights[0]) * (max_year - min_year)
    print("Q8a: " + str(x_star))
    print("Q8b: This prediction assumes a linear trend, which may not hold true in reality as environmental factors may cause non-linear changes.")

# Main function to execute the entire workflow
if __name__ == '__main__':
    # Read command-line arguments
    filename = sys.argv[1]
    learning_rate = float(sys.argv[2])
    iterations = int(sys.argv[3])

    # Load the data
    years, frozen_days = load_data(filename)

    # Visualize the data (Q2)
    visualize_data(years, frozen_days)

    # Normalize the data (Q3)
    X_normalized, min_year, max_year = normalize_data(years)

    # Closed-form solution for linear regression (Q4)
    weights_closed_form = closed_form_solution(X_normalized, frozen_days)

    # Gradient descent for linear regression (Q5)
    theta_gd, cost_history = gradient_descent(X_normalized, frozen_days, learning_rate, iterations)

    # Save the cost function plot (Q5d)
    plt.plot(range(iterations), cost_history)
    plt.xlabel("Iterations")
    plt.ylabel("Cost (MSE)")
    plt.title("Cost Function during Gradient Descent")
    plt.grid(True)
    plt.savefig("loss_plot.jpg")

    # Prediction for 2023-24 using closed-form solution (Q6)
    predict_for_2023(weights_closed_form, min_year, max_year)

    # Model interpretation based on the sign of the slope (Q7)
    model_interpretation(weights_closed_form)

    # Model limitation: Year Lake Mendota will no longer freeze (Q8)
    model_limitation(weights_closed_form, min_year, max_year)
