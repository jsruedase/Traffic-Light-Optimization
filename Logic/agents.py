import collections
import random

class TrafficAgent:
    def __init__(self, epsilon: float, gamma: float, alpha: float):
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.weights = collections.Counter()
        
    def getFeatures(self, state, action):
        ns_green, ns_cars, we_cars, ns_weight, we_weight, max_time_green = state
        features = collections.Counter()    
        
        if action == "switch":
            next_ns_green = not ns_green
        else:
            next_ns_green = ns_green
            
        features["bias"] = 1.0
        
        if next_ns_green:
            features["active_lane_cars"] = ns_cars/100
            features["inactive_lane_cars"] = we_cars/100
        else:
            features["active_lane_cars"] = we_cars/100
            features["inactive_lane_cars"] = ns_cars/100
            

        if action == "switch":
            if max_time_green < 3:
                features["switch_very_fast"] = 1.0  
            elif max_time_green < 5:
                features["switch_fast"] = 1.0 
            elif max_time_green < 8:
                features["switch_moderate"] = 1.0 
        
        if action == "switch" and max_time_green > 0:
            features["switch_inversely_proportional"] = 10.0 / (max_time_green + 1)
        
        if action == "stay" and max_time_green < 5:
            features["patience_reward"] = 1.0

        if next_ns_green:
            features["active_lane_eagerness"] = ns_weight/100
            features["inactive_lane_eagerness"] = we_weight/100
        else:
            features["active_lane_eagerness"] = we_weight/100
            features["inactive_lane_eagerness"] = ns_weight/100
            
        return features
        
    def getQValue(self, state, action):
        features = self.getFeatures(state, action)
        q_value = 0
        for feature, value in features.items():
            q_value += self.weights[feature] * value
        return q_value
    
    def computeValueFromQValues(self, state):
        return max(self.getQValue(state, action) for action in ["switch", "stay"])
    
    def computeActionFromQValues(self, state):
        best_action = None
        best_value = float('-inf')
        for action in ["switch", "stay"]:
            q_value = self.getQValue(state, action)
            if q_value > best_value:
                best_value = q_value
                best_action = action
        return best_action
    
    def getAction(self, state):
        if random.random() < self.epsilon:
            return random.choice(["switch", "stay"])
        else:
            return self.computeActionFromQValues(state)
        
    def update(self, state, action, nextState, reward):
        features = self.getFeatures(state, action)
        q_value = self.getQValue(state, action)
        next_value = self.computeValueFromQValues(nextState)
        difference = (reward + self.gamma * next_value) - q_value
        
        for feature, value in features.items():
            self.weights[feature] += self.alpha * difference * value

class NaiveAgent:
    """Agente que cambia el semáforo cada N pasos fijos"""
    def __init__(self, switch_interval: int):
        self.switch_interval = switch_interval
        self.steps_since_switch = 0
        
    def getAction(self, state):
        self.steps_since_switch += 1
        if self.steps_since_switch >= self.switch_interval:
            self.steps_since_switch = 0
            return "switch"
        return "stay"