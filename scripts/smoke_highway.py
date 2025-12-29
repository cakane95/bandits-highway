#./scripts/smoke_highway.py

"""
    Script pour tester la configuration de l'environnement `highway-v0`
"""
import gymnasium as gym
import highway_env

# Initialiser l'environnement (sans rendu graphique)
env = gym.make("highway-v0", render_mode="rgb_array")

MAX_STEP_SIZE = 5

# Faire un reset et générer la première observation
observation, info = env.reset()

# Affichage
print(f"Espace d'actions : {env.action_space}")
print(f"Type d'observation : {type(observation)}")
print(f"Dimensions de l'observation : {observation.shape}")
print(f"Informations supplémentaires : {info}")

episode_over = False
total_reward = 0.0

for step in range(MAX_STEP_SIZE):
    # Sélectionner une action aléatoire
    action = env.action_space.sample()

    # Appliquer l'action
    observation, reward, terminated, truncated, info = env.step(action)

    # Affichage
    print(f"[Step {step}] Action choisie : {action}")
    print(f"[Step {step}] Récompense : {reward}")

    total_reward += reward
    episode_over = terminated or truncated

    if episode_over:
        print("Fin de l’épisode")
        break

print(f"Récompense cumulée sur l'épisode : {total_reward}")
env.close()