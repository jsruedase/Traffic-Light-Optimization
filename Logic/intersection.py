import random
import numpy as np

class Car:
    def __init__(self, orientation: str, eagerness: int = None, eagerness_distribution: str = "poisson"):
        # Diferentes distribuciones para el eagerness
        if eagerness is not None:
            self.eagerness = eagerness
        elif eagerness_distribution == "poisson":
            # Poisson con lambda=2 (mayoría 1-3, raramente >5)
            self.eagerness = min(np.random.poisson(2) + 1, 10)
        elif eagerness_distribution == "exponential":
            # Exponencial truncada (mayoría bajos, algunos muy altos)
            self.eagerness = min(int(np.random.exponential(2)) + 1, 10)
        elif eagerness_distribution == "beta":
            # Beta(2,5) sesgada hacia valores bajos
            self.eagerness = max(1, int(np.random.beta(2, 5) * 10))
        elif eagerness_distribution == "normal_low":
            # Normal con media baja (μ=3, σ=1.5)
            self.eagerness = max(1, min(10, int(np.random.normal(3, 1.5))))
        else:  # "uniform" (original)
            self.eagerness = random.randint(1, 10)
            
        self.orientation = orientation
        self.wait_time = 0

class TrafficLight:
    def __init__(self, orientation: str):
        self.orientation = orientation
        self.is_green = False
        self.time_green = 0
    
    def switch(self):
        self.is_green = not self.is_green
        if not self.is_green:
            self.time_green = 0
            
    def update_time(self):
        if self.is_green:
            self.time_green += 1

class Intersection:
    def __init__(self, eagerness_distribution: str = "poisson"):
        self.ns_traffic_light = TrafficLight("NS")
        self.we_traffic_light = TrafficLight("WE")
        self.ns_cars = []
        self.we_cars = []
        self.eagerness_distribution = eagerness_distribution
        
    def add_car(self):
        p = random.uniform(0,1)
        q = random.uniform(0,1)
        if p < 0.5:
            self.ns_cars.append(Car("NS", eagerness_distribution=self.eagerness_distribution))
        
        if q < 0.2:
            self.we_cars.append(Car("WE", eagerness_distribution=self.eagerness_distribution))

    def getState(self):
        return (
            self.ns_traffic_light.is_green, 
            len(self.ns_cars), 
            len(self.we_cars), 
            sum(car.eagerness for car in self.ns_cars),
            sum(car.eagerness for car in self.we_cars),
            max(self.ns_traffic_light.time_green, self.we_traffic_light.time_green)
        )

    def step(self, action: str):
        if action == "switch":
            self.ns_traffic_light.switch()
            self.we_traffic_light.switch()
        
        self.ns_traffic_light.update_time()
        self.we_traffic_light.update_time()
        
        self.add_car()
        
        # Incrementar tiempo de espera de todos los carros
        for car in self.ns_cars:
            car.wait_time += 1
        for car in self.we_cars:
            car.wait_time += 1
        
        # Dejar pasar carros y registrar su tiempo de espera
        cars_passed_wait_time = 0
        if self.ns_traffic_light.is_green and self.ns_cars:
            car = self.ns_cars.pop(0)
            cars_passed_wait_time = car.wait_time
        if self.we_traffic_light.is_green and self.we_cars:
            car = self.we_cars.pop(0)
            cars_passed_wait_time = car.wait_time
            
        # Penalización por espera (usando afán)
        wait_penalty = 0
        if self.ns_traffic_light.is_green:
            wait_penalty = sum(car.eagerness for car in self.we_cars)
        else:
            wait_penalty = sum(car.eagerness for car in self.ns_cars)
            
        return self.getState(), -wait_penalty, cars_passed_wait_time