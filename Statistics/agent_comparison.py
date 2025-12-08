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
    """Entrena el agente de RL y registra métricas de aprendizaje"""
    agent = TrafficAgent(epsilon=0.1, gamma=0.9, alpha=0.01)
    
    # Métricas de entrenamiento
    episode_queues = []  # Cola promedio por episodio
    episode_rewards = []  # Recompensa total por episodio
    
    for episode in range(num_episodes):
        intersection = Intersection(eagerness_distribution=eagerness_dist)
        state = intersection.getState()
        
        episode_reward = 0
        queues_in_episode = []
        
        for step in range(max_steps_per_episode):
            action = agent.getAction(state)
            nextState, reward, _ = intersection.step(action)
            agent.update(state, action, nextState, reward)
            
            episode_reward += reward
            total_queue = len(intersection.ns_cars) + len(intersection.we_cars)
            queues_in_episode.append(total_queue)
            
            state = nextState
        
        # Registrar métricas del episodio
        episode_queues.append(np.mean(queues_in_episode))
        episode_rewards.append(episode_reward)
    
    return agent, episode_queues, episode_rewards

def compare_agents(num_episodes: int = 1000, max_steps_per_episode: int = 500):
    """Entrena y compara diferentes agentes"""
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
    
    # Entrenar agentes RL con cada distribución
    rl_agents = {}
    learning_curves = {}  # Guardar curvas de aprendizaje
    
    for eagerness_dist in distributions:
        print(f"=== ENTRENANDO AGENTE RL CON DISTRIBUCIÓN: {eagerness_dist.upper()} ===")
        rl_agent, queues, rewards = train_rl_agent(num_episodes=num_episodes, max_steps_per_episode=max_steps_per_episode, eagerness_dist=eagerness_dist)
        print("Pesos aprendidos:", rl_agent.weights)
        print()
        rl_agents[eagerness_dist] = rl_agent
        learning_curves[eagerness_dist] = {'queues': queues, 'rewards': rewards}
    
    # Crear gráficas de convergencia
    print("=== GENERANDO GRÁFICAS DE CONVERGENCIA ===\n")
    
    # Gráfica 1: Cola promedio durante entrenamiento
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    for dist in distributions:
        queues = learning_curves[dist]['queues']
        # Suavizar con ventana móvil
        window_size = 50
        smoothed_queues = np.convolve(queues, np.ones(window_size)/window_size, mode='valid')
        axes[0].plot(smoothed_queues, label=f'{dist.capitalize()}', linewidth=2, alpha=0.8)
    
    # Encontrar el mejor promedio final
    best_dist = min(distributions, key=lambda d: np.mean(learning_curves[d]['queues'][-100:]))
    best_avg = np.mean(learning_curves[best_dist]['queues'][-100:])
    
    axes[0].axhline(y=best_avg, color='cyan', linestyle='--', linewidth=3, 
                    label=f'Mejor promedio: {best_avg:.3f}')
    axes[0].set_xlabel('# Episodio', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Carros promedio en cola', fontsize=12, fontweight='bold')
    axes[0].set_title('Carros promedio en cola por episodio (suavizado)', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Gráfica 2: Recompensa durante entrenamiento
    for dist in distributions:
        rewards = learning_curves[dist]['rewards']
        # Suavizar con ventana móvil
        smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        axes[1].plot(smoothed_rewards, label=f'{dist.capitalize()}', linewidth=2, alpha=0.8)
    
    # Mejor recompensa final
    best_reward_dist = max(distributions, key=lambda d: np.mean(learning_curves[d]['rewards'][-100:]))
    best_reward_avg = np.mean(learning_curves[best_reward_dist]['rewards'][-100:])
    
    axes[1].axhline(y=best_reward_avg, color='cyan', linestyle='--', linewidth=3, 
                    label=f'Mejor promedio: {best_reward_avg:.1f}')
    axes[1].set_xlabel('# Episodio', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Recompensa acumulada', fontsize=12, fontweight='bold')
    axes[1].set_title('Recompensa por episodio (suavizado)', fontsize=14, fontweight='bold')
    axes[1].legend(loc='lower right', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Statistics/Graphs/learning_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Curvas de aprendizaje guardadas en 'Statistics/Graphs/learning_curves.png'\n")
    
    # Crear agentes naive (estos son independientes de la distribución)
    naive_agents = [
        (NaiveAgent(5), "Naive Agent (5 pasos)"),
        (NaiveAgent(10), "Naive Agent (10 pasos)"),
        (NaiveAgent(15), "Naive Agent (15 pasos)"),
        (NaiveAgent(20), "Naive Agent (20 pasos)")
    ]
    
    print("=== EVALUANDO AGENTES ===")
    results = []
    
    # Evaluar cada agente RL con SU distribución correspondiente
    for eagerness_dist in distributions:
        agent = rl_agents[eagerness_dist]
        name = f"RL Agent ({eagerness_dist.capitalize()})"
        print(f"Evaluando {name}...")
        result = evaluate_agent(agent, num_episodes=100, max_steps_per_episode=max_steps_per_episode, 
                                agent_name=name, eagerness_dist=eagerness_dist)
        results.append(result)
    
    # Evaluar agentes naive con distribución uniform (o elige una por defecto)
    eval_dist = "uniform"  # Puedes cambiar esto
    for agent, name in naive_agents:
        print(f"Evaluando {name}...")
        result = evaluate_agent(agent, num_episodes=100, max_steps_per_episode=max_steps_per_episode, 
                                agent_name=name, eagerness_dist=eval_dist)
        results.append(result)
    
    # Mostrar resultados
    print("\n=== RESULTADOS COMPARATIVOS ===\n")
    print(f"{'Agente':<35} {'Recompensa Avg':<15} {'Cola Avg':<12} {'Cola Max':<12} {'Tiempo Espera':<15} {'Cambios Avg':<12}")
    print("-" * 115)
    
    for r in results:
        print(f"{r['name']:<35} {r['avg_reward']:>12.2f}  {r['avg_queue']:>10.2f}  {r['max_queue']:>10.2f}  {r['avg_wait_time']:>13.2f}  {r['avg_switches']:>10.2f}")
    
    # Crear visualizaciones
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Comparación de Agentes de Control de Semáforos', fontsize=16)
    
    # Gráfica 1: Recompensa promedio
    names = [r['name'] for r in results]
    rewards = [r['avg_reward'] for r in results]
    colors = ['green' if 'RL' in name else 'orange' for name in names]
    
    axes[0, 0].bar(range(len(names)), rewards, color=colors)
    axes[0, 0].set_xticks(range(len(names)))
    axes[0, 0].set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    axes[0, 0].set_ylabel('Recompensa Promedio')
    axes[0, 0].set_title('Recompensa Total (Mayor es Mejor)')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Gráfica 2: Longitud promedio de cola
    queues = [r['avg_queue'] for r in results]
    axes[0, 1].bar(range(len(names)), queues, color=colors)
    axes[0, 1].set_xticks(range(len(names)))
    axes[0, 1].set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    axes[0, 1].set_ylabel('Carros en Cola')
    axes[0, 1].set_title('Longitud Promedio de Cola (Menor es Mejor)')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Gráfica 3: Tiempo de espera promedio
    wait_times = [r['avg_wait_time'] for r in results]
    axes[1, 0].bar(range(len(names)), wait_times, color=colors)
    axes[1, 0].set_xticks(range(len(names)))
    axes[1, 0].set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    axes[1, 0].set_ylabel('Pasos de Espera')
    axes[1, 0].set_title('Tiempo Promedio de Espera (Menor es Mejor)')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Gráfica 4: Número de cambios de semáforo
    switches = [r['avg_switches'] for r in results]
    axes[1, 1].bar(range(len(names)), switches, color=colors)
    axes[1, 1].set_xticks(range(len(names)))
    axes[1, 1].set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    axes[1, 1].set_ylabel('Número de Cambios')
    axes[1, 1].set_title('Cambios de Semáforo Promedio')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Statistics/Graphs/traffic_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Gráficas guardadas en 'Statistics/Graphs/traffic_comparison.png'")
    
    # Calcular mejora porcentual de cada RL vs el mejor Naive
    print("\n=== MEJORA DE AGENTES RL VS MEJOR NAIVE ===\n")
    naive_results = [r for r in results if 'Naive' in r['name']]
    best_naive_reward = max(r['avg_reward'] for r in naive_results)
    
    for r in results:
        if 'RL' in r['name']:
            improvement = ((r['avg_reward'] - best_naive_reward) / abs(best_naive_reward)) * 100
            print(f"{r['name']:<35}: {improvement:>6.2f}% mejor que el mejor Naive")
    
    # Análisis de penalizaciones por cambio rápido
    print(f"\n=== ANÁLISIS DE FEATURES APRENDIDOS ===\n")
    for dist, agent in rl_agents.items():
        print(f"\n{dist.upper()}:")
        switch_features = ['switch_very_fast', 'switch_fast', 'switch_moderate', 
                           'switch_inversely_proportional', 'patience_reward']
        for feature in switch_features:
            weight = agent.weights.get(feature, 0)
            if weight != 0:
                print(f"  {feature:.<35} {weight:>10.4f}")

if __name__ == "__main__":
    random.seed(42)  # Para reproducibilidad
    compare_agents(num_episodes=1000, max_steps_per_episode=500)