import csv
import random
import numpy as np
from math import radians, sin, cos, sqrt, atan2

class TSP_GA:
    def __init__(self, num_nodes, num_population, num_generations, mutation_rate, distances):
        self.num_nodes = num_nodes
        self.num_population = num_population
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.distances = distances

    def create_initial_population(self):
        # Generate random tours as the initial population
        population = []
        for _ in range(self.num_population):
            tour = list(range(1, self.num_nodes))  # Exclude the starting node (depot)
            random.shuffle(tour)
            population.append([0] + tour)  # Include the starting node (depot) at the beginning
        return population

    def evaluate_fitness(self, population):
        # Calculate the total distance for each tour in the population
        fitness = []
        for tour in population:
            total_distance = sum(self.distances[tour[i - 1]][tour[i]] for i in range(1, self.num_nodes))
            fitness.append(total_distance)
        return fitness

    def selection(self, population, fitness):
        # Roulette wheel selection
        total_fitness = sum(fitness)
        if total_fitness == 0:
            probabilities = [1 / len(population)] * len(population)  # Equal probabilities if total_fitness is zero
        else:
            probabilities = [fit / total_fitness for fit in fitness]
        selected_indices = np.random.choice(len(population), size=self.num_population, p=probabilities)
        return [population[i] for i in selected_indices]

    def crossover(self, parent1, parent2):
        # Order crossover (OX1)
        num_nodes = self.num_nodes - 1  # Exclude the starting node
        start = random.randint(1, max(1, num_nodes - 1))
        end = random.randint(start + 1, num_nodes)
        offspring = [-1] * num_nodes
        offspring[start:end] = parent1[start:end]
        remaining = [city for city in parent2 if city not in offspring[start:end]]
        offspring[:start] = remaining[:start]
        offspring[end:] = remaining[start:]
        return offspring

    def mutation(self, tour):
        # Swap mutation
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(1, self.num_nodes), 2)
            tour[idx1], tour[idx2] = tour[idx2], tour[idx1]
        return tour

    def run(self):
        population = self.create_initial_population()
        for _ in range(self.num_generations):
            fitness = self.evaluate_fitness(population)
            population = self.selection(population, fitness)
            next_generation = []
            while len(next_generation) < self.num_population:
                parent1, parent2 = random.sample(population, 2)
                offspring = self.crossover(parent1, parent2)
                offspring = self.mutation(offspring)
                next_generation.append(offspring)
            population = next_generation
        best_tour = min(population, key=lambda x: sum(self.distances[x[i - 1]][x[i]] for i in range(1, self.num_nodes)))
        return best_tour

def haversine_distance(lat1, lon1, lat2, lon2):
    # Calculate the haversine distance between two points given their latitude and longitude
    R = 6371.0  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

def parse_csv(filename):
    depot = None
    customers = []

    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            order_id = int(row[0])
            lat, lon = float(row[1]), float(row[2])
            depot_lat, depot_lon = float(row[3]), float(row[4])
            if depot is None:
                depot = (depot_lat, depot_lon)
            customers.append((order_id, lat, lon))

    return depot, customers

def calculate_distances(depot, customers):
    num_nodes = len(customers)
    distances_matrix = []
    for i in range(num_nodes):
        row = []
        for j in range(num_nodes):
            if i == j:
                row.append(0)  # Distance from a node to itself is 0
            else:
                lat1, lon1 = customers[i][1], customers[i][2]
                lat2, lon2 = customers[j][1], customers[j][2]
                dist = haversine_distance(lat1, lon1, lat2, lon2)
                row.append(dist)
        distances_matrix.append(row)
    return distances_matrix





if __name__ == "__main__":
    filename = "/home/prasanjith/Music/sct_hackathon_round_1/input_datasets/part_a/part_a_input_dataset_5.csv"  # Adjust the filename as needed
    depot, customers = parse_csv(filename)
    distances = calculate_distances(depot, customers)
    

    # GA parameters
    num_nodes = len(customers)
    num_population = 10000
    num_generations = 100
    mutation_rate = 0.2

    # Initialize GA and run
    tsp_ga = TSP_GA(num_nodes, num_population, num_generations, mutation_rate, distances)
    best_tour = tsp_ga.run()
    total_distance_km = sum(distances[best_tour[i - 1]][best_tour[i]] for i in range(1, len(best_tour)))
    total_distance_km += distances[best_tour[-1]][0]

    # Print total distance of the best tour in kilometers
    print("Total distance of the best tour:", total_distance_km, "kilometers")
