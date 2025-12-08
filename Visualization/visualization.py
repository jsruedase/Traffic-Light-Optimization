from Logic.intersection import Intersection, Car
from Logic.agents import NaiveAgent, TrafficAgent
import numpy as np
import matplotlib.pyplot as plt
import random
import tkinter as tk
from tkinter import ttk


class TrafficVisualization:
    def __init__(self, root):
        self.root = root
        self.root.title("Visualización de Semáforo Inteligente")
        self.root.geometry("1200x800")
        
        # Variables de control
        self.running = False
        self.speed = 200  # ms entre frames
        self.current_agent = None
        self.intersection = None
        self.step_count = 0
        self.total_reward = 0
        self.reward_history = []
        self.queue_history = []
        
        self.setup_ui()
        
    def setup_ui(self):
        # Frame superior: Controles
        control_frame = tk.Frame(self.root, bg="#2c3e50", height=80)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(control_frame, text="Agente:", bg="#2c3e50", fg="white", font=("Arial", 10)).grid(row=0, column=0, padx=5, pady=5)
        self.agent_type = ttk.Combobox(control_frame, values=["RL Agent", "Naive (5 pasos)", "Naive (10 pasos)", "Naive (20 pasos)"], state="readonly", width=18)
        self.agent_type.set("RL Agent")
        self.agent_type.grid(row=0, column=1, padx=5, pady=5)
        
        tk.Label(control_frame, text="Distribución:", bg="#2c3e50", fg="white", font=("Arial", 10)).grid(row=0, column=2, padx=5, pady=5)
        self.dist_type = ttk.Combobox(control_frame, values=["uniform", "poisson", "exponential", "beta", "normal_low"], state="readonly", width=12)
        self.dist_type.set("uniform")
        self.dist_type.grid(row=0, column=3, padx=5, pady=5)
        
        tk.Label(control_frame, text="Velocidad:", bg="#2c3e50", fg="white", font=("Arial", 10)).grid(row=0, column=4, padx=5, pady=5)
        self.speed_scale = tk.Scale(control_frame, from_=50, to=500, orient=tk.HORIZONTAL, command=self.update_speed, bg="#34495e", fg="white", highlightthickness=0, length=100)
        self.speed_scale.set(200)
        self.speed_scale.grid(row=0, column=5, padx=5, pady=5)
        
        self.start_button = tk.Button(control_frame, text="▶ Iniciar", command=self.start_simulation, bg="#27ae60", fg="white", font=("Arial", 10, "bold"), width=8)
        self.start_button.grid(row=0, column=6, padx=5, pady=5)
        
        self.stop_button = tk.Button(control_frame, text="⏸ Pausar", command=self.stop_simulation, bg="#e74c3c", fg="white", font=("Arial", 10, "bold"), width=8, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=7, padx=5, pady=5)
        
        self.reset_button = tk.Button(control_frame, text="🔄 Reset", command=self.reset_simulation, bg="#3498db", fg="white", font=("Arial", 10, "bold"), width=8)
        self.reset_button.grid(row=0, column=8, padx=5, pady=5)
        
        # Frame central: Canvas (MÁS GRANDE)
        self.canvas = tk.Canvas(self.root, bg="#ecf0f1", width=1180, height=620)
        self.canvas.pack(padx=10, pady=5)
        
        # Frame inferior: Estadísticas (MÁS COMPACTO)
        stats_frame = tk.Frame(self.root, bg="#34495e", height=80)
        stats_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Stats labels en una sola fila compacta
        self.step_label = tk.Label(stats_frame, text="Paso: 0", bg="#34495e", fg="white", font=("Arial", 11, "bold"))
        self.step_label.grid(row=0, column=0, padx=15, pady=5)
        
        self.reward_label = tk.Label(stats_frame, text="Recompensa: 0", bg="#34495e", fg="white", font=("Arial", 11, "bold"))
        self.reward_label.grid(row=0, column=1, padx=15, pady=5)
        
        self.ns_queue_label = tk.Label(stats_frame, text="Cola N-S: 0 (peso: 0)", bg="#34495e", fg="#3498db", font=("Arial", 10))
        self.ns_queue_label.grid(row=0, column=2, padx=15, pady=5)
        
        self.we_queue_label = tk.Label(stats_frame, text="Cola W-E: 0 (peso: 0)", bg="#34495e", fg="#e67e22", font=("Arial", 10))
        self.we_queue_label.grid(row=0, column=3, padx=15, pady=5)
        
        self.time_label = tk.Label(stats_frame, text="Tiempo Verde: 0", bg="#34495e", fg="#9b59b6", font=("Arial", 10))
        self.time_label.grid(row=0, column=4, padx=15, pady=5)
        
    def update_speed(self, value):
        self.speed = int(value)
        
    def start_simulation(self, num_episodes: int = 1000, max_steps_per_episode: int = 500):
        if not self.running:
            self.running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            
            # Inicializar agente e intersección
            agent_choice = self.agent_type.get()
            dist = self.dist_type.get()
            
            if agent_choice == "RL Agent":
                # Entrenar rápidamente
                self.current_agent = TrafficAgent(epsilon=0.1, gamma=0.9, alpha=0.01)
                print("Entrenando agente RL...")
                for ep in range(num_episodes):
                    temp_intersection = Intersection(eagerness_distribution=dist)
                    state = temp_intersection.getState()
                    for _ in range(max_steps_per_episode):
                        action = self.current_agent.getAction(state)
                        nextState, reward, _ = temp_intersection.step(action)
                        self.current_agent.update(state, action, nextState, reward)
                        state = nextState
                print("Entrenamiento completo!")
            elif "5 pasos" in agent_choice:
                self.current_agent = NaiveAgent(5)
            elif "10 pasos" in agent_choice:
                self.current_agent = NaiveAgent(10)
            elif "20 pasos" in agent_choice:
                self.current_agent = NaiveAgent(20)
            
            self.intersection = Intersection(eagerness_distribution=dist)
            self.intersection.ns_traffic_light.is_green = True
            self.step_count = 0
            self.total_reward = 0
            self.reward_history = []
            self.queue_history = []
            
            self.animate()
    
    def stop_simulation(self):
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
    
    def reset_simulation(self):
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.canvas.delete("all")
        self.step_count = 0
        self.total_reward = 0
        self.reward_history = []
        self.queue_history = []
        self.update_stats()
    
    def animate(self):
        if self.running:
            state = self.intersection.getState()
            action = self.current_agent.getAction(state)
            nextState, reward, _ = self.intersection.step(action)
            
            self.step_count += 1
            self.total_reward += reward
            self.reward_history.append(self.total_reward)
            self.queue_history.append(len(self.intersection.ns_cars) + len(self.intersection.we_cars))
            
            self.draw_intersection(action)
            self.update_stats()
            
            self.root.after(self.speed, self.animate)
    
    def draw_intersection(self, last_action):
        self.canvas.delete("all")
        
        canvas_width = 1180
        canvas_height = 620
        center_x = canvas_width // 2
        center_y = canvas_height // 2
        
        # Dibujar calles
        # Calle horizontal (W-E)
        self.canvas.create_rectangle(0, center_y - 60, canvas_width, center_y + 60, fill="#7f8c8d", outline="")
        self.canvas.create_line(0, center_y, canvas_width, center_y, fill="white", width=2, dash=(10, 10))
        
        # Calle vertical (N-S)
        self.canvas.create_rectangle(center_x - 60, 0, center_x + 60, canvas_height, fill="#7f8c8d", outline="")
        self.canvas.create_line(center_x, 0, center_x, canvas_height, fill="white", width=2, dash=(10, 10))
        
        # Dibujar semáforos
        ns_color = "#27ae60" if self.intersection.ns_traffic_light.is_green else "#e74c3c"
        we_color = "#27ae60" if self.intersection.we_traffic_light.is_green else "#e74c3c"
        
        # Semáforo N-S (arriba)
        self.canvas.create_oval(center_x - 20, center_y - 150, center_x + 20, center_y - 110, fill=ns_color, outline="black", width=3)
        self.canvas.create_text(center_x, center_y - 170, text=f"N-S: {self.intersection.ns_traffic_light.time_green}s", font=("Arial", 10, "bold"))
        
        # Semáforo W-E (izquierda)
        self.canvas.create_oval(center_x - 150, center_y - 20, center_x - 110, center_y + 20, fill=we_color, outline="black", width=3)
        self.canvas.create_text(center_x - 170, center_y, text=f"W-E: {self.intersection.we_traffic_light.time_green}s", font=("Arial", 10, "bold"), angle=90)
        
        # Dibujar carros N-S (desde arriba) - Mostrar hasta 30 carros
        car_size = 14
        spacing = 20
        for i, car in enumerate(self.intersection.ns_cars[:30]):
            y_pos = center_y - 180 - (i * spacing)
            color = self.get_color_by_eagerness(car.eagerness)
            self.canvas.create_rectangle(center_x - car_size, y_pos - car_size, 
                                        center_x + car_size, y_pos + car_size, 
                                        fill=color, outline="black", width=2)
            self.canvas.create_text(center_x, y_pos, text=str(car.eagerness), font=("Arial", 8, "bold"), fill="white")
        
        if len(self.intersection.ns_cars) > 30:
            self.canvas.create_text(center_x, 15, text=f"↑ +{len(self.intersection.ns_cars) - 30} más", 
                                   font=("Arial", 12, "bold"), fill="#e74c3c")
        
        # Dibujar carros W-E (desde izquierda) - Mostrar hasta 30 carros
        for i, car in enumerate(self.intersection.we_cars[:30]):
            x_pos = center_x - 180 - (i * spacing)
            color = self.get_color_by_eagerness(car.eagerness)
            self.canvas.create_rectangle(x_pos - car_size, center_y - car_size, 
                                        x_pos + car_size, center_y + car_size, 
                                        fill=color, outline="black", width=2)
            self.canvas.create_text(x_pos, center_y, text=str(car.eagerness), font=("Arial", 8, "bold"), fill="white")
        
        if len(self.intersection.we_cars) > 30:
            self.canvas.create_text(15, center_y - 35, text=f"← +{len(self.intersection.we_cars) - 30}", 
                                   font=("Arial", 12, "bold"), fill="#e74c3c")
        
        # Indicador de última acción
        if last_action == "switch":
            self.canvas.create_text(center_x, 30, text="⚠ CAMBIO DE SEMÁFORO", font=("Arial", 18, "bold"), fill="#e74c3c")
    
    def get_color_by_eagerness(self, eagerness):
        # Verde (bajo afán) a Rojo (alto afán)
        ratio = eagerness / 10.0
        r = int(255 * ratio)
        g = int(255 * (1 - ratio))
        return f'#{r:02x}{g:02x}00'
    
    def update_stats(self):
        self.step_label.config(text=f"Paso: {self.step_count}")
        self.reward_label.config(text=f"Recompensa: {self.total_reward:.1f}")
        
        ns_weight = sum(car.eagerness for car in self.intersection.ns_cars) if self.intersection else 0
        we_weight = sum(car.eagerness for car in self.intersection.we_cars) if self.intersection else 0
        
        self.ns_queue_label.config(text=f"Cola N-S: {len(self.intersection.ns_cars) if self.intersection else 0} (peso: {ns_weight})")
        self.we_queue_label.config(text=f"Cola W-E: {len(self.intersection.we_cars) if self.intersection else 0} (peso: {we_weight})")
        
        max_time = max(self.intersection.ns_traffic_light.time_green, self.intersection.we_traffic_light.time_green) if self.intersection else 0
        self.time_label.config(text=f"Tiempo Verde: {max_time}")

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficVisualization(root)
    root.mainloop()

