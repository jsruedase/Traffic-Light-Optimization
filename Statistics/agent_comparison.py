from Logic.intersection import Intersection, Car
from Logic.agents import NaiveAgent, TrafficAgent
import numpy as np
import matplotlib.pyplot as plt
import random

def evaluate_agent(agent, num_episodes: int, max_steps_per_episode: int, agent_name: str, eagerness_dist: str = "poisson"):
    """Evalúa un agente y retorna métricas de desempeño"""
    total_rewards = []
    avg_queue_lengths = []
    max_queue_lengths = []
    avg_wait_times = []
    switches_count = []
    
    for episode in range(num_episodes):
        intersection = Intersection(eagerness_distribution=eagerness_dist)
        state = intersection.getState()
        episode_reward = 0
        episode_queues = []
        episode_switches = 0
        wait_times = []
        
        for step in range(max_steps_per_episode):
            action = agent.getAction(state)
            if action == "switch":
                episode_switches += 1
                
            nextState, reward, wait_time = intersection.step(action)
            episode_reward += reward
            
            total_queue = len(intersection.ns_cars) + len(intersection.we_cars)
            episode_queues.append(total_queue)
            
            if wait_time > 0:
                wait_times.append(wait_time)
            
            state = nextState
            
        total_rewards.append(episode_reward)
        avg_queue_lengths.append(np.mean(episode_queues))
        max_queue_lengths.append(np.max(episode_queues))
        switches_count.append(episode_switches)
        if wait_times:
            avg_wait_times.append(np.mean(wait_times))
    
    return {
        'name': agent_name,
        'avg_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'avg_queue': np.mean(avg_queue_lengths),
        'max_queue': np.mean(max_queue_lengths),
        'avg_wait_time': np.mean(avg_wait_times) if avg_wait_times else 0,
        'avg_switches': np.mean(switches_count),
        'all_rewards': total_rewards,
        'all_queues': avg_queue_lengths
    }

def train_rl_agent(num_episodes: int, max_steps_per_episode: int, eagerness_dist: str = "poisson"):
    """Entrena el agente de RL"""
    agent = TrafficAgent(epsilon=0.1, gamma=0.9, alpha=0.01)
    
    for episode in range(num_episodes):
        intersection = Intersection(eagerness_distribution=eagerness_dist)
        state = intersection.getState()
        
        for step in range(max_steps_per_episode):
            action = agent.getAction(state)
            nextState, reward, _ = intersection.step(action)
            agent.update(state, action, nextState, reward)
            state = nextState
    
    return agent

def compare_agents():
    """Entrena y compara diferentes agentes"""
    # Primero, comparar diferentes distribuciones de eagerness
    distributions = ["uniform", "poisson", "exponential", "beta", "normal_low"]
    
    print("=== ANÁLISIS DE DISTRIBUCIONES DE EAGERNESS ===\n")
    
    # Generar muestra de cada distribución
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Distribuciones de Eagerness (Afán de los Carros)', fontsize=16)
    
    for idx, dist in enumerate(distributions):
        sample = [Car("NS", eagerness_distribution=dist).eagerness for _ in range(1000)]
        
        ax = axes[idx // 3, idx % 3]
        ax.hist(sample, bins=range(1, 12), alpha=0.7, edgecolor='black')
        ax.set_title(f'{dist.capitalize()}\nμ={np.mean(sample):.2f}, σ={np.std(sample):.2f}')
        ax.set_xlabel('Nivel de Afán')
        ax.set_ylabel('Frecuencia')
        ax.set_xlim(0, 11)
        ax.grid(axis='y', alpha=0.3)
    
    # Ocultar el último subplot vacío
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('Statistics/Graphs/eagerness_distributions.png', dpi=300, bbox_inches='tight')
    print("✓ Distribuciones guardadas en 'Statistics/Graphs/eagerness_distributions.png'\n")
    
    # Entrenar con la distribución Poisson (recomendada)

    # Crear agentes naive con diferentes intervalos
    agents = [
        (NaiveAgent(5), "Naive Agent (5 pasos)"),
        (NaiveAgent(10), "Naive Agent (10 pasos)"),
        (NaiveAgent(15), "Naive Agent (15 pasos)"),
        (NaiveAgent(20), "Naive Agent (20 pasos)")
    ]
    

    for eagerness_dist in distributions:
        print(f"=== ENTRENANDO AGENTE RL CON DISTRIBUCIÓN: {eagerness_dist.upper()} ===")
        rl_agent = train_rl_agent(num_episodes=100, max_steps_per_episode=2000, eagerness_dist=eagerness_dist)
        print("Pesos aprendidos:", rl_agent.weights)
        print()
        agents.insert(0, (rl_agent, f"RL Agent ({eagerness_dist.capitalize()})"))
    

    print("=== EVALUANDO AGENTES ===")
    results = []
    for agent, name in agents:
        print(f"Evaluando {name}...")
        result = evaluate_agent(agent, num_episodes=100, max_steps_per_episode=500, 
                                agent_name=name, eagerness_dist=eagerness_dist)
        results.append(result)
    
    # Mostrar resultados
    print("\n=== RESULTADOS COMPARATIVOS ===\n")
    print(f"{'Agente':<30} {'Recompensa Avg':<15} {'Cola Avg':<12} {'Cola Max':<12} {'Tiempo Espera':<15} {'Cambios Avg':<12}")
    print("-" * 110)
    
    for r in results:
        print(f"{r['name']:<30} {r['avg_reward']:>12.2f}  {r['avg_queue']:>10.2f}  {r['max_queue']:>10.2f}  {r['avg_wait_time']:>13.2f}  {r['avg_switches']:>10.2f}")
    
    # Crear visualizaciones
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Comparación de Agentes (Distribución: {eagerness_dist.capitalize()})', fontsize=16)
    
    # Gráfica 1: Recompensa promedio
    names = [r['name'] for r in results]
    rewards = [r['avg_reward'] for r in results]
    colors = ['green' if 'RL' in name else 'orange' for name in names]
    
    axes[0, 0].bar(range(len(names)), rewards, color=colors)
    axes[0, 0].set_xticks(range(len(names)))
    axes[0, 0].set_xticklabels(names, rotation=45, ha='right')
    axes[0, 0].set_ylabel('Recompensa Promedio')
    axes[0, 0].set_title('Recompensa Total (Mayor es Mejor)')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Gráfica 2: Longitud promedio de cola
    queues = [r['avg_queue'] for r in results]
    axes[0, 1].bar(range(len(names)), queues, color=colors)
    axes[0, 1].set_xticks(range(len(names)))
    axes[0, 1].set_xticklabels(names, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Carros en Cola')
    axes[0, 1].set_title('Longitud Promedio de Cola (Menor es Mejor)')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Gráfica 3: Tiempo de espera promedio
    wait_times = [r['avg_wait_time'] for r in results]
    axes[1, 0].bar(range(len(names)), wait_times, color=colors)
    axes[1, 0].set_xticks(range(len(names)))
    axes[1, 0].set_xticklabels(names, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Pasos de Espera')
    axes[1, 0].set_title('Tiempo Promedio de Espera (Menor es Mejor)')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Gráfica 4: Distribución de recompensas
    for r in results:
        axes[1, 1].hist(r['all_rewards'], alpha=0.5, label=r['name'], bins=20)
    axes[1, 1].set_xlabel('Recompensa por Episodio')
    axes[1, 1].set_ylabel('Frecuencia')
    axes[1, 1].set_title('Distribución de Recompensas')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Statistics/Graphs/traffic_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Gráficas guardadas en 'Statistics/Graphs/traffic_comparison.png'")

if __name__ == "__main__":
    random.seed(42)  # Para reproducibilidad
    compare_agents()