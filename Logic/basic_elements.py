import random
import time

class Car:
    """
    El objetivo es que cada carro tenga ciertos atributos gráficos y ciertos que afecten
    su comportamiento al cruzar semáforos, principalmente que tenga afán o no
    """
    def __init__(self, orientation: str, eagerness: int = None):
        self.color = random.choice(["red, white, black, green, blue"])
        s = random.uniform(0,1)
        self.size = 1 if s< 0.6 else 2 if s< 0.9 else 3
        self.eagerness = eagerness if eagerness is not None else random.randint(1,10) #Afán, entre mayor hará que el peso de la fila incremente
        self.orientation = orientation
            

class TrafficLight:
    """
    Manejará el sentido de los carros y si está dando paso
    """
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
    def __init__(self):
        self.ns_traffic_light = TrafficLight("NS")
        self.we_traffic_light = TrafficLight("WE")
        self.ns_cars = []
        self.we_cars = []
        
    def add_car(self):
        p = random.uniform(0,1)
        q = random.uniform(0,1)
        if p < 0.7:
            self.ns_cars.append(Car("NS"))
        
        if q < 0.5:
            self.we_cars.append(Car("WE"))
    
    def naive_update_traffic_lights(self):
        self.ns_traffic_light.update_time()
        self.we_traffic_light.update_time()
        
        if self.ns_traffic_light.is_green and self.ns_traffic_light.time_green >= 10:
            self.ns_traffic_light.switch()
            self.we_traffic_light.switch()
        elif self.we_traffic_light.is_green and self.we_traffic_light.time_green >= 10:
            self.we_traffic_light.switch()
            self.ns_traffic_light.switch()
        elif not self.ns_traffic_light.is_green and not self.we_traffic_light.is_green:
            self.ns_traffic_light.switch()

    def calculate_ns_weight(self):
        return sum(car.eagerness for car in self.ns_cars)
    
    def calculate_we_weight(self):
        return sum(car.eagerness for car in self.we_cars)
    
    def total_weight(self):
        return self.calculate_ns_weight() + self.calculate_we_weight()
    
    def naive_simulation(self, duration: int, verbose: bool = False):
        for _ in range(duration):
            self.add_car()
            self.naive_update_traffic_lights()
            time.sleep(1)  # Simulate time passing (optional)
            if verbose:
                print(f"NS Light: {'Green' if intersection.ns_traffic_light.is_green else 'Red'}, "
                f"WE Light: {'Green' if intersection.we_traffic_light.is_green else 'Red'}, "
                f"NS Cars: {len(intersection.ns_cars)}, WE Cars: {len(intersection.we_cars)}"
                f" Total Weight: {intersection.total_weight()}")
            if intersection.ns_traffic_light.is_green:
                intersection.ns_cars.pop(0) if intersection.ns_cars else None
            if intersection.we_traffic_light.is_green:
                intersection.we_cars.pop(0) if intersection.we_cars else None

if __name__ == "__main__":
    intersection = Intersection()
    intersection.naive_simulation(duration=60, verbose=True)
            
    